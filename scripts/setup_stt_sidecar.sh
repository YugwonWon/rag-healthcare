#!/bin/bash
# STT 사이드카용 격리 venv 구성 (호스트 mlx-whisper, Apple Silicon Metal GPU).
# 컨테이너 CPU(faster-whisper)보다 훨씬 빠른 STT. melo 사이드카와 동일한 구조.
#
# 사용: ./scripts/setup_stt_sidecar.sh [venv경로]   (기본: <repo>/.venv-stt)

set -e

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${1:-$PROJECT/.venv-stt}"
PY311="$(command -v python3.11 || echo /opt/homebrew/bin/python3.11)"

echo "▶ Python: $PY311 ($($PY311 --version 2>&1))"
echo "▶ venv 생성: $VENV"
"$PY311" -m venv "$VENV"
PIP="$VENV/bin/pip"
"$PIP" install -q -U pip wheel setuptools
echo "▶ mlx-whisper + 서버 의존성 설치"
"$PIP" install mlx-whisper fastapi "uvicorn[standard]" python-multipart numpy

echo "✅ 완료: $VENV"
echo "   실행: ./scripts/run_stt_sidecar.sh"
