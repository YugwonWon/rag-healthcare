#!/bin/bash
# Cloud Run용 시작 스크립트: Ollama + Qwen + FastAPI

set -e

echo "🚀 Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Ollama 서버가 준비될 때까지 대기
echo "⏳ Waiting for Ollama to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "❌ Ollama failed to start"
        exit 1
    fi
    sleep 1
done

# 모델 확인 및 다운로드
MODEL_NAME="${OLLAMA_MODEL:-qwen2.5:3b}"
echo "📦 Checking model: ${MODEL_NAME}..."

if ! ollama list | grep -q "${MODEL_NAME}"; then
    echo "⬇️ Pulling model: ${MODEL_NAME} (this may take a while on first run)..."
    ollama pull ${MODEL_NAME}
    echo "✅ Model pulled successfully!"
else
    echo "✅ Model already available!"
fi

# 모델 미리 로드 (워밍업)
echo "🔥 Warming up model..."
curl -s http://localhost:11434/api/generate -d "{\"model\": \"${MODEL_NAME}\", \"prompt\": \"hello\", \"stream\": false}" > /dev/null 2>&1 || true
echo "✅ Model warmed up!"

# FastAPI 앱 실행
echo "🌐 Starting FastAPI server on port ${PORT:-8000}..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
