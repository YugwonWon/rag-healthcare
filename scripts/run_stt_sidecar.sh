#!/bin/bash
# STT 사이드카 실행 (호스트 mlx-whisper, Metal GPU).
# 메인 앱은 settings.STT_SIDECAR_URL(기본 http://127.0.0.1:8182/transcribe)로 호출.
# 사전: ./scripts/setup_stt_sidecar.sh 로 .venv-stt 구성 완료.

set -e

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${STT_VENV:-$PROJECT/.venv-stt}"
export STT_PORT="${STT_PORT:-8182}"
export STT_LANGUAGE="${STT_LANGUAGE:-ko}"
export STT_MLX_REPO="${STT_MLX_REPO:-mlx-community/whisper-large-v3-turbo}"

if [ ! -x "$VENV/bin/python" ]; then
    echo "❌ $VENV 가 없습니다. 먼저 ./scripts/setup_stt_sidecar.sh 를 실행하세요."
    exit 1
fi

echo "▶ STT 사이드카 시작: http://127.0.0.1:${STT_PORT} (mlx=${STT_MLX_REPO})"
exec "$VENV/bin/python" "$PROJECT/scripts/stt_sidecar_server.py"
