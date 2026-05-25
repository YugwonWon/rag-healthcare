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
import wave
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

MODEL_REPO = os.getenv("STT_MLX_REPO", "mlx-community/whisper-large-v3-turbo")
LANGUAGE = os.getenv("STT_LANGUAGE", "ko")
PORT = int(os.getenv("STT_PORT", "8182"))

app = FastAPI(title="STT sidecar (mlx-whisper)")
_loaded = False


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
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


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
    result = mlx_whisper.transcribe(
        arr, path_or_hf_repo=MODEL_REPO, language=language or LANGUAGE
    )
    return {"text": (result.get("text") or "").strip()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
