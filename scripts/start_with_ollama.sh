#!/bin/bash
# Cloud Run용 시작 스크립트: Ollama + FastAPI (범용 모델 지원)
# 모델은 Dockerfile에서 이미 pre-registered되므로 런타임에서는 서버 시작만 필요

# set -e 사용하지 않음 - 부분 실패에도 서버는 시작해야 함

export PYTHONIOENCODING=utf-8
export OLLAMA_DEBUG=0

# ─── 1. Ollama 서버 시작 ───
echo "🚀 Starting Ollama server..."
ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!

echo "⏳ Waiting for Ollama to be ready..."
OLLAMA_READY=false
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is ready! (${i}s)"
        OLLAMA_READY=true
        break
    fi
    sleep 1
done

if [ "$OLLAMA_READY" = false ]; then
    echo "⚠️ Ollama not ready yet, but starting server anyway..."
fi

# ─── 2. 모델 확인 (Dockerfile에서 이미 pre-registered) ───
MODEL_NAME="${OLLAMA_MODEL:-k-exaone-counseling}"
echo "📦 Model: ${MODEL_NAME}"

if [ "$OLLAMA_READY" = true ]; then
    if ollama list 2>/dev/null | grep -q "${MODEL_NAME}"; then
        echo "✅ Model already registered (pre-built)!"
    else
        echo "⚠️ Model not found, attempting registration..."
        MODELS_DIR="/app/models"
        MODELFILE="${MODELS_DIR}/Modelfile.${MODEL_NAME}"
        GGUF_FILE="${MODELS_DIR}/${MODEL_NAME}.gguf"
        if [ -f "${MODELFILE}" ]; then
            cd "${MODELS_DIR}" && ollama create "${MODEL_NAME}" -f "Modelfile.${MODEL_NAME}" 2>&1 && cd /app || echo "⚠️ Model create failed, continuing..."
        elif [ -f "${GGUF_FILE}" ]; then
            printf "FROM ${GGUF_FILE}\nPARAMETER temperature 0.1\n" > /tmp/Modelfile.auto
            ollama create "${MODEL_NAME}" -f /tmp/Modelfile.auto 2>&1 || echo "⚠️ Model create failed, continuing..."
            rm -f /tmp/Modelfile.auto
        else
            ollama pull "${MODEL_NAME}" 2>&1 || echo "⚠️ Model pull failed, continuing..."
        fi
    fi
else
    echo "⚠️ Ollama not ready, skipping model check"
fi

# ─── 3. 서버 시작 후 백그라운드 워밍업 ───
echo "🔥 Warmup will run after server starts..."

# ─── 4. 디렉토리 준비 ───
mkdir -p /app/data/chroma /app/logs

# ─── 5. FastAPI 서버 실행 ───
PORT=${PORT:-8080}
echo "🌐 Starting FastAPI server on port ${PORT}..."
echo "   Model: ${MODEL_NAME}"

# 서버 시작 후 백그라운드에서 워밍업 + 데이터 초기화
(
    sleep 10
    # 워밍업
    if [ "$OLLAMA_READY" = true ]; then
        curl -s http://localhost:11434/api/generate \
            -d "{\"model\": \"${MODEL_NAME}\", \"prompt\": \"hello\", \"stream\": false}" > /dev/null 2>&1 || true
        echo "✅ Background warmup complete!"
    fi
) &

exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
