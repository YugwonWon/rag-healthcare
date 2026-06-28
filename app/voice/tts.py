"""음성 합성(TTS) — 다중 엔진 지원 (지연 로딩).

엔진(설정 TTS_ENGINE):
- edge : Microsoft Edge 온라인 TTS. 한국어 자연성 최상, 설치 간단(`pip install edge-tts`).
         단 합성 텍스트(응답)가 MS로 전송됨(환자 음성은 아님). 모델 다운로드 없음.
- say  : macOS 내장 TTS(기본 보이스 Yuna). 완전 온디바이스·무설치. 품질 보통.
- melo : MeloTTS(한국어), 온디바이스 고품질. 설치 까다로움
         (`pip install git+https://github.com/myshell-ai/MeloTTS.git` + g2pkk).

synthesize()는 (audio_bytes, media_type) 튜플을 반환한다.
"""

import asyncio
import concurrent.futures
import os
import subprocess
import tempfile
from typing import Optional, Tuple

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


def _run_coro_blocking(coro):
    """실행 중인 이벤트 루프 여부와 무관하게 코루틴을 동기적으로 실행한다.

    FastAPI(async) 안에서 호출돼도 안전하도록, 루프가 돌고 있으면 별도 스레드의
    새 루프에서 실행한다."""
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is not None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ── edge-tts (MS Edge 온라인) ──
def _synth_edge(text: str, speed: float) -> Tuple[bytes, str]:
    import edge_tts  # 지연 로딩
    voice = settings.TTS_VOICE or "ko-KR-SunHiNeural"
    # speed(1.0=기본)를 edge rate 문자열로: 0.9 → "-10%"
    rate = f"{int(round((speed - 1.0) * 100)):+d}%"
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        path = f.name

    async def _go():
        await edge_tts.Communicate(text, voice, rate=rate).save(path)

    try:
        _run_coro_blocking(_go())
        with open(path, "rb") as fh:
            audio = fh.read()
        logger.info(f"TTS(edge) 합성 완료 | voice={voice} | bytes={len(audio)}")
        return audio, "audio/mpeg"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ── macOS say (온디바이스) ──
def _synth_say(text: str, speed: float) -> Tuple[bytes, str]:
    voice = settings.TTS_VOICE or "Yuna"
    # say -r 은 분당 단어수(기본 ~175). speed 0.9 → 약 158
    rate = max(90, int(175 * speed))
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
        aiff = f.name
    wav = aiff[:-5] + ".wav"
    try:
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", aiff, text], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", "-c", "1", aiff, wav],
            check=True,
        )
        with open(wav, "rb") as fh:
            audio = fh.read()
        logger.info(f"TTS(say) 합성 완료 | voice={voice} | bytes={len(audio)}")
        return audio, "audio/wav"
    finally:
        for p in (aiff, wav):
            try:
                os.remove(p)
            except OSError:
                pass


# ── MeloTTS (온디바이스 고품질) — 별도 .venv-tts 사이드카에 HTTP 호출 ──
# melo는 transformers 4.27 핀 + mecab 충돌로 메인 venv와 공존 불가 → 별도 프로세스로 띄운다.
# scripts/setup_melo_tts.sh 로 .venv-tts 구성 후 scripts/run_melo_sidecar.sh 로 실행.
def _downsample_wav(wav_bytes: bytes, target_sr: int = 16000) -> bytes:
    """MeloTTS의 44.1kHz WAV을 16kHz mono로 다운샘플한다.

    프론트는 음성 답변을 base64 data URL 로 Gradio Textbox 를 통해 전달하는데,
    44.1kHz WAV 은 응답이 조금만 길어도 2MB 를 넘겨 전달 중 손상→자동재생 실패가
    관찰됨(2026-06). 16kHz 광대역은 음성 명료도 손실이 사실상 없으면서 크기를 ~2.7배
    줄여 data URL 을 안정권으로 낮춘다. 실패 시 원본을 그대로 반환한다."""
    import audioop
    import io
    import wave
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            ch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
            frames = w.readframes(w.getnframes())
        if sr == target_sr and ch == 1 and sw == 2:
            return wav_bytes
        if sw != 2:
            frames = audioop.lin2lin(frames, sw, 2); sw = 2
        if ch == 2:
            frames = audioop.tomono(frames, 2, 0.5, 0.5); ch = 1
        if sr != target_sr:
            frames, _ = audioop.ratecv(frames, 2, 1, sr, target_sr, None)
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(target_sr)
            w.writeframes(frames)
        return out.getvalue()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"TTS 다운샘플 실패(원본 반환): {e}")
        return wav_bytes


def _synth_melo(text: str, speed: float) -> Tuple[bytes, str]:
    import httpx
    url = settings.MELO_TTS_URL
    try:
        resp = httpx.post(url, json={"text": text, "speed": speed}, timeout=settings.TTS_TIMEOUT)
        resp.raise_for_status()
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"MeloTTS 사이드카에 연결할 수 없습니다({url}). "
            "'./scripts/run_melo_sidecar.sh'로 사이드카를 먼저 실행하세요."
        ) from e
    audio = _downsample_wav(resp.content)
    logger.info(f"TTS(melo, sidecar) 합성 완료 | bytes={len(resp.content)}→{len(audio)} (16kHz)")
    return audio, "audio/wav"


def synthesize(text: str, speed: Optional[float] = None) -> Tuple[bytes, str]:
    """텍스트를 음성으로 합성한다. (audio_bytes, media_type) 반환."""
    engine = settings.TTS_ENGINE
    spd = speed if speed is not None else settings.TTS_SPEED
    if engine == "edge":
        return _synth_edge(text, spd)
    if engine == "say":
        return _synth_say(text, spd)
    if engine == "melo":
        return _synth_melo(text, spd)
    raise RuntimeError(f"지원하지 않는 TTS 엔진: {engine}")
