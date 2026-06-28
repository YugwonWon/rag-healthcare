"""음성 인식(STT) — 온디바이스 Whisper.

기본 엔진은 faster-whisper(이식성: Mac CPU + Cloud Run linux 모두 동작).
Mac GPU 최속을 원하면 STT_ENGINE=mlx-whisper 로 전환(Apple Silicon 전용).
모델은 최초 호출 시 1회 로딩되어 싱글톤으로 재사용된다.
"""

import os
from typing import Optional

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

# faster-whisper 모델 싱글톤
_fw_model = None


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    # faster-whisper(ctranslate2)는 Metal을 지원하지 않으므로 Mac에서는 cpu(int8)가 안정적.
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_faster_whisper():
    global _fw_model
    if _fw_model is None:
        from faster_whisper import WhisperModel  # 지연 로딩
        device = _resolve_device(settings.STT_DEVICE)
        compute_type = settings.STT_COMPUTE_TYPE
        logger.info(
            f"STT 모델 로딩(faster-whisper): {settings.STT_MODEL} "
            f"| device={device} | compute_type={compute_type}"
        )
        _fw_model = WhisperModel(
            settings.STT_MODEL, device=device, compute_type=compute_type
        )
    return _fw_model


def _transcribe_sidecar(audio_path: str, language: str) -> str:
    """호스트 STT 사이드카(mlx-whisper, Metal GPU)에 오디오를 보내 전사한다."""
    import httpx
    url = settings.STT_SIDECAR_URL
    try:
        with open(audio_path, "rb") as f:
            files = {"audio": (os.path.basename(audio_path), f, "audio/wav")}
            resp = httpx.post(url, files=files, data={"language": language}, timeout=settings.STT_TIMEOUT)
        resp.raise_for_status()
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"STT 사이드카에 연결할 수 없습니다({url}). "
            "'./scripts/run_stt_sidecar.sh'로 사이드카를 먼저 실행하세요."
        ) from e
    text = (resp.json().get("text") or "").strip()
    logger.info(f"STT 전사(sidecar) 완료 | lang={language} | len={len(text)}")
    return text


def transcribe(audio_path: str, language: Optional[str] = None) -> str:
    """오디오 파일을 텍스트로 전사한다.

    Args:
        audio_path: 전사할 오디오 파일 경로(wav/webm/mp3 등 ffmpeg 디코딩 가능 포맷).
        language: 언어 코드(기본: settings.STT_LANGUAGE = "ko").

    Returns:
        전사된 텍스트(공백 정리됨). 인식 실패 시 빈 문자열.
    """
    lang = language or settings.STT_LANGUAGE
    engine = settings.STT_ENGINE

    if engine == "sidecar":
        return _transcribe_sidecar(audio_path, lang)

    if engine == "mlx-whisper":
        import mlx_whisper  # 지연 로딩 (Apple Silicon 전용)
        repo = settings.STT_MLX_REPO
        logger.info(f"STT 전사(mlx-whisper): repo={repo} | lang={lang}")
        result = mlx_whisper.transcribe(
            audio_path, path_or_hf_repo=repo, language=lang
        )
        return (result.get("text") or "").strip()

    # 기본: faster-whisper
    model = _get_faster_whisper()
    # vad_filter로 무음 구간을 제거해 정확도/속도를 높인다.
    segments, info = model.transcribe(audio_path, language=lang, vad_filter=True)
    text = "".join(seg.text for seg in segments).strip()
    logger.info(f"STT 전사 완료 | lang={lang} | len={len(text)}")
    return text


def warmup() -> bool:
    """STT 모델을 미리 로드(필요 시 다운로드)한다.

    컨테이너 startup에서 호출해 첫 사용자 요청의 콜드스타트(모델 다운로드+로드) 지연을 없앤다.
    무음 더미 입력으로 VAD 필터까지 한 번 돌려 전체 경로를 초기화한다."""
    if settings.STT_ENGINE in ("sidecar", "mlx-whisper"):
        # 사이드카는 자체 startup에서 워밍업 / mlx는 파일 기반 로드 → 컨테이너 워밍업 불필요
        return True
    try:
        import numpy as np
        model = _get_faster_whisper()
        segments, _ = model.transcribe(
            np.zeros(16000, dtype=np.float32), language=settings.STT_LANGUAGE, vad_filter=True
        )
        list(segments)  # 생성자 소비 → 실제 초기화 완료
        logger.info("STT 워밍업 완료")
        return True
    except Exception as e:
        logger.warning(f"STT 워밍업 실패(첫 요청 시 재시도): {e}")
        return False
