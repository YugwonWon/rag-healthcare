"""MeloTTS 한국어 TTS 사이드카 서버 (별도 .venv-tts에서 실행되는 standalone 스크립트).

메인 앱과 다른 venv(py3.11, transformers 4.27, patched melo)에서 동작한다.
모델을 1회 로드해 warm 상태로 유지하고, 메인 앱이 HTTP로 POST /synth 호출한다.
메인 앱 패키지(app.*)를 import하지 않는 self-contained 스크립트다.

실행: scripts/run_melo_sidecar.sh  (또는 .venv-tts/bin/python scripts/melo_sidecar_server.py)
환경변수: MELO_PORT(기본 8181), MELO_LANGUAGE(KR), MELO_SPEED(0.9), MELO_DEVICE(auto)
"""

import os
import tempfile
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

LANGUAGE = os.getenv("MELO_LANGUAGE", "KR")
DEFAULT_SPEED = float(os.getenv("MELO_SPEED", "0.9"))
DEVICE = os.getenv("MELO_DEVICE", "auto")
PORT = int(os.getenv("MELO_PORT", "8181"))

app = FastAPI(title="MeloTTS sidecar")

_tts = None
_speaker_id = None


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _get_tts():
    global _tts, _speaker_id
    if _tts is None:
        from melo.api import TTS
        device = _resolve_device(DEVICE)
        print(f"[melo-sidecar] 모델 로딩: {LANGUAGE} | device={device}", flush=True)
        _tts = TTS(language=LANGUAGE, device=device)
        spk2id = _tts.hps.data.spk2id
        # 한국어 단일 화자: 첫 화자 id 사용 (spk2id 값이 화자 인덱스)
        _speaker_id = list(spk2id.values())[0]
        print(f"[melo-sidecar] 모델 준비 완료 | spk2id={dict(spk2id)} | speaker_id={_speaker_id}", flush=True)
    return _tts, _speaker_id


class SynthRequest(BaseModel):
    text: str
    speed: Optional[float] = None


@app.on_event("startup")
def _warmup():
    try:
        _get_tts()  # 첫 요청 지연을 줄이기 위해 미리 로드
    except Exception as e:  # noqa: BLE001
        print(f"[melo-sidecar] warmup 실패(첫 요청 시 재시도): {e}", flush=True)


@app.get("/health")
def health():
    return {"ok": True, "language": LANGUAGE, "loaded": _tts is not None}


@app.post("/synth")
def synth(req: SynthRequest):
    tts, speaker_id = _get_tts()
    speed = req.speed if req.speed is not None else DEFAULT_SPEED
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name
    try:
        tts.tts_to_file(req.text, speaker_id, out_path, speed=speed)
        with open(out_path, "rb") as fh:
            data = fh.read()
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass
    return Response(content=data, media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
