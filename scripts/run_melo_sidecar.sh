#!/bin/bash
# MeloTTS 한국어 TTS 사이드카 실행 (별도 .venv-tts).
# 메인 앱은 settings.MELO_TTS_URL(기본 http://127.0.0.1:8181/synth)로 이 서버를 호출한다.
#
# 사전: ./scripts/setup_melo_tts.sh 로 .venv-tts 구성 완료.
# 사용: ./scripts/run_melo_sidecar.sh

set -e

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${MELO_VENV:-$PROJECT/.venv-tts}"
export MELO_PORT="${MELO_PORT:-8181}"
export MELO_LANGUAGE="${MELO_LANGUAGE:-KR}"
export MELO_SPEED="${MELO_SPEED:-0.9}"
export MELO_DEVICE="${MELO_DEVICE:-auto}"
export PATH="/opt/homebrew/bin:$PATH"   # mecab-config (런타임)

if [ ! -x "$VENV/bin/python" ]; then
    echo "❌ $VENV 가 없습니다. 먼저 ./scripts/setup_melo_tts.sh 를 실행하세요."
    exit 1
fi

echo "▶ MeloTTS 사이드카 시작: http://127.0.0.1:${MELO_PORT} (venv=$VENV)"
exec "$VENV/bin/python" "$PROJECT/scripts/melo_sidecar_server.py"
