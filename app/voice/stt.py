"""음성 인식(STT) — 온디바이스 Whisper.

기본 엔진은 faster-whisper(이식성: Mac CPU + Cloud Run linux 모두 동작).
Mac GPU 최속을 원하면 STT_ENGINE=mlx-whisper 로 전환(Apple Silicon 전용).
모델은 최초 호출 시 1회 로딩되어 싱글톤으로 재사용된다.
"""

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

    if engine == "mlx-whisper":
        import mlx_whisper  # 지연 로딩 (Apple Silicon 전용)
        repo = f"mlx-community/whisper-{settings.STT_MODEL}"
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
