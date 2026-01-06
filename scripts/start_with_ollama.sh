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
    # 3번 재시도
    for attempt in 1 2 3; do
        if ollama pull ${MODEL_NAME}; then
            echo "✅ Model pulled successfully!"
            break
        else
            echo "⚠️ Pull attempt $attempt failed, retrying..."
            sleep 5
        fi
        if [ $attempt -eq 3 ]; then
            echo "❌ Failed to pull model after 3 attempts"
            exit 1
        fi
    done
else
    echo "✅ Model already available!"
fi

# 모델 검증 (한글 테스트)
echo "🔍 Verifying model with Korean test..."
KOREAN_TEST=$(curl -s http://localhost:11434/api/generate -d "{\"model\": \"${MODEL_NAME}\", \"prompt\": \"안녕하세요라고 말해주세요\", \"stream\": false}" 2>&1)
if echo "$KOREAN_TEST" | grep -q "안녕"; then
    echo "✅ Korean language support verified!"
else
    echo "⚠️ Korean test response: $KOREAN_TEST"
    echo "⚠️ Model may have issues with Korean, but continuing..."
fi

# 모델 미리 로드 (워밍업)
echo "🔥 Warming up model..."
curl -s http://localhost:11434/api/generate -d "{\"model\": \"${MODEL_NAME}\", \"prompt\": \"hello\", \"stream\": false}" > /dev/null 2>&1 || true
echo "✅ Model warmed up!"

# ChromaDB 상태 확인 및 문서 초기화
echo "📚 Checking ChromaDB data..."
if [ -d "/app/data/chroma" ]; then
    echo "✅ ChromaDB directory exists"
    ls -la /app/data/chroma/ || true
else
    echo "⚠️ ChromaDB directory not found, creating..."
    mkdir -p /app/data/chroma
fi

# 문서 수 확인 및 초기화 (Python으로)
echo "📄 Checking document count..."
python3 -c "
from pathlib import Path
from app.vector_store import get_chroma_handler
chroma = get_chroma_handler()
stats = chroma.get_collection_stats()
print(f'Documents: {stats[\"documents\"]}')
print(f'Conversations: {stats[\"conversations\"]}')
print(f'Profiles: {stats[\"patient_profiles\"]}')

if stats['documents'] == 0:
    print('⚠️ No documents found, loading healthcare docs...')
    import sys
    sys.path.insert(0, '/app')
    from scripts.load_healthcare_docs import load_all_documents
    docs_dir = Path('/app/data/healthcare_docs')
    if docs_dir.exists():
        load_all_documents(docs_dir)
        # 다시 확인
        stats = chroma.get_collection_stats()
        print(f'After loading - Documents: {stats[\"documents\"]}')
    else:
        print(f'⚠️ Healthcare docs directory not found: {docs_dir}')
else:
    print('✅ Documents already loaded')
"

# FastAPI 앱 실행
echo "🌐 Starting FastAPI server on port ${PORT:-8000}..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
