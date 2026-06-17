"""STT 사이드카 서버 (별도 .venv-stt에서 실행되는 standalone 스크립트).

호스트 네이티브로 실행돼 Apple Silicon Metal GPU(mlx-whisper)를 사용한다.
컨테이너 CPU(faster-whisper) 대비 훨씬 빠르다. 모델을 1회 로드(warm)하고 메인 앱이
HTTP로 POST /transcribe 호출한다. 메인 앱 패키지(app.*)를 import하지 않는다.

입력 오디오는 WAV로 가정하고 stdlib(wave/audioop)로 16kHz mono float32 변환 → ffmpeg 불필요.
실행: scripts/run_stt_sidecar.sh  (또는 .venv-stt/bin/python scripts/stt_sidecar_server.py)
환경변수: STT_PORT(8182), STT_MLX_REPO, STT_LANGUAGE(ko)
"""

import audioop
import io
import os
import re
import wave
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

MODEL_REPO = os.getenv("STT_MLX_REPO", "mlx-community/whisper-large-v3-turbo")
LANGUAGE = os.getenv("STT_LANGUAGE", "ko")
PORT = int(os.getenv("STT_PORT", "8182"))
# 긴 발화 안정화: 한 번에 처리할 오디오 최대 길이(초). 어르신이 쉼 없이 길게
# 말씀하시면 Whisper 환각·지연이 급증하므로 앞부분만 받아 처리한다.
MAX_AUDIO_SEC = float(os.getenv("STT_MAX_AUDIO_SEC", "30"))

app = FastAPI(title="STT sidecar (mlx-whisper)")
_loaded = False


def _collapse_repeats(text: str) -> str:
    """Whisper 환각으로 같은 토막이 반복되는 것을 접어준다.

    예) "처음엔 처음엔 처음엔 ... 처음" → "처음엔". 무음/잡음 구간에서
    동일 단어·구가 수십 번 반복 생성되는 알려진 현상을 후처리로 제거한다.
    """
    if not text:
        return text
    # 1) 공백 기준 연속 동일 토큰 축약 ("처음엔 처음엔 처음엔" → "처음엔")
    tokens = text.split()
    deduped: list[str] = []
    for tok in tokens:
        if deduped and deduped[-1] == tok:
            continue
        deduped.append(tok)
    out = " ".join(deduped)
    # 2) 동일 구(2~10자)가 3회 이상 연속 반복되면 1회로 ("아이고아이고아이고" 류)
    out = re.sub(r"(.{2,10}?)\1{2,}", r"\1", out)
    # 3) 같은 글자 4회 이상 연속 → 1회 ("아아아아아" → "아")
    out = re.sub(r"(.)\1{3,}", r"\1", out)
    return out.strip()


def _decode_wav_to_f32_16k(data: bytes) -> "np.ndarray":
    """WAV 바이트 → 16kHz mono float32 (ffmpeg 없이 stdlib로)."""
    with wave.open(io.BytesIO(data), "rb") as w:
        ch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        frames = w.readframes(w.getnframes())
    if sw != 2:  # 16-bit PCM으로 정규화
        frames = audioop.lin2lin(frames, sw, 2)
        sw = 2
    if ch == 2:  # 모노로
        frames = audioop.tomono(frames, 2, 0.5, 0.5)
    if sr != 16000:  # 16kHz로 리샘플
        frames, _ = audioop.ratecv(frames, 2, 1, sr, 16000, None)
    arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    # 긴 발화 안정화: 최대 길이를 초과하면 앞부분만 사용
    max_samples = int(MAX_AUDIO_SEC * 16000)
    if max_samples and arr.shape[0] > max_samples:
        print(
            f"[stt-sidecar] 오디오 {arr.shape[0]/16000:.1f}s → "
            f"{MAX_AUDIO_SEC:.0f}s로 절단(환각·지연 방지)",
            flush=True,
        )
        arr = arr[:max_samples]
    return arr


def _warm():
    global _loaded
    import mlx_whisper
    print(f"[stt-sidecar] 모델 로딩: {MODEL_REPO}", flush=True)
    mlx_whisper.transcribe(
        np.zeros(16000, dtype=np.float32), path_or_hf_repo=MODEL_REPO, language=LANGUAGE
    )
    _loaded = True
    print("[stt-sidecar] 모델 준비 완료", flush=True)


@app.on_event("startup")
def _startup():
    try:
        _warm()
    except Exception as e:  # noqa: BLE001
        print(f"[stt-sidecar] warmup 실패(첫 요청 시 재시도): {e}", flush=True)


@app.get("/health")
def health():
    return {"ok": True, "loaded": _loaded, "repo": MODEL_REPO}


@app.post("/transcribe")
def transcribe(audio: UploadFile = File(...), language: Optional[str] = Form(None)):
    import mlx_whisper
    data = audio.file.read()
    arr = _decode_wav_to_f32_16k(data)
    # 환각(반복) 억제 디코딩 옵션:
    #  - condition_on_previous_text=False: 직전 생성 텍스트를 다음 윈도우 컨텍스트로
    #    되먹이지 않음 → "처음엔 처음엔 ..." 식 반복 루프의 핵심 원인 차단.
    #  - no_speech_threshold/logprob/compression: 무음·저신뢰 구간을 묵음 처리.
    #  - temperature 폴백: 디코딩 실패 시 점진적으로 온도를 올려 재시도.
    result = mlx_whisper.transcribe(
        arr,
        path_or_hf_repo=MODEL_REPO,
        language=language or LANGUAGE,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        temperature=(0.0, 0.2, 0.4, 0.6),
    )
    text = _collapse_repeats((result.get("text") or "").strip())
    return {"text": text}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
