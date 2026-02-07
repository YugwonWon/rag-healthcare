#!/bin/bash
# Cloud Run용 시작 스크립트: Ollama + FastAPI (범용 모델 지원)
#
# 모델 등록 방식:
#   1. models/{모델명}.gguf + models/Modelfile.{모델명} 이 있으면 → 자동 등록
#   2. Modelfile만 있으면 → Modelfile로 등록 (GGUF 경로가 Modelfile 안에 지정)
#   3. 둘 다 없으면 → ollama pull로 다운로드 시도

set -e

export PYTHONIOENCODING=utf-8
export OLLAMA_DEBUG=0

# ─── 1. Ollama 서버 시작 ───
echo "🚀 Starting Ollama server..."
ollama serve 2>&1 | grep -v "print_info\|llama_\|ggml_\|rope_\|vocab\|token" &
OLLAMA_PID=$!

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

# ─── 2. 모델 등록 (범용) ───
MODEL_NAME="${OLLAMA_MODEL:-k-exaone-counseling}"
MODELS_DIR="/app/models"
GGUF_FILE="${MODELS_DIR}/${MODEL_NAME}.gguf"
MODELFILE="${MODELS_DIR}/Modelfile.${MODEL_NAME}"

echo "📦 Model: ${MODEL_NAME}"
echo "   GGUF:      ${GGUF_FILE}"
echo "   Modelfile:  ${MODELFILE}"

if ollama list 2>/dev/null | grep -q "${MODEL_NAME}"; then
    echo "✅ Model already registered!"
else
    if [ -f "${MODELFILE}" ]; then
        # Modelfile이 있으면 사용
        echo "📝 Registering model with Modelfile..."
        ollama create "${MODEL_NAME}" -f "${MODELFILE}"
        echo "✅ ${MODEL_NAME} registered!"
    elif [ -f "${GGUF_FILE}" ]; then
        # GGUF만 있으면 기본 Modelfile 자동 생성
        echo "📝 Generating default Modelfile for ${MODEL_NAME}..."
        cat > /tmp/Modelfile.auto << EOF
FROM ${GGUF_FILE}
SYSTEM "당신은 노인건강전문상담사입니다. 3~4문장으로 간결하게 답변하세요."
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 512
PARAMETER num_ctx 4096
EOF
        ollama create "${MODEL_NAME}" -f /tmp/Modelfile.auto
        rm -f /tmp/Modelfile.auto
        echo "✅ ${MODEL_NAME} registered (auto-generated Modelfile)!"
    else
        # 둘 다 없으면 Ollama Hub에서 pull
        echo "⬇️ No local files found. Pulling from Ollama Hub: ${MODEL_NAME}..."
        for attempt in 1 2 3; do
            if ollama pull "${MODEL_NAME}" 2>&1 | tail -5; then
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
    fi
fi

# ─── 3. 모델 검증 (한글 테스트) ───
echo "🔍 Verifying model..."
KOREAN_TEST=$(curl -s http://localhost:11434/api/generate \
    -d "{\"model\": \"${MODEL_NAME}\", \"prompt\": \"안녕하세요라고 말해주세요\", \"stream\": false}" 2>&1)
if echo "$KOREAN_TEST" | grep -q "안녕"; then
    echo "✅ Korean language support verified!"
else
    echo "⚠️ Model may have issues with Korean, but continuing..."
fi

# 워밍업
echo "🔥 Warming up model..."
curl -s http://localhost:11434/api/generate \
    -d "{\"model\": \"${MODEL_NAME}\", \"prompt\": \"hello\", \"stream\": false}" > /dev/null 2>&1 || true
echo "✅ Model ready!"

# ─── 4. 데이터 초기화 ───
echo "📚 Checking data store..."
if [ -d "/app/data/chroma" ]; then
    echo "✅ ChromaDB directory exists"
else
    echo "⚠️ ChromaDB directory not found, creating..."
    mkdir -p /app/data/chroma
fi

echo "📄 Checking and loading documents..."
python3 -c "
from pathlib import Path
from app.vector_store import get_chroma_handler
chroma = get_chroma_handler()
stats = chroma.get_collection_stats()
print(f'Current - Documents: {stats[\"documents\"]}')
print(f'          Conversations: {stats[\"conversations\"]}')
print(f'          Profiles: {stats[\"patient_profiles\"]}')

import sys
sys.path.insert(0, '/app')
docs_dir = Path('/app/data/healthcare_docs')

if docs_dir.exists():
    doc_files = list(docs_dir.glob('*.txt')) + list(docs_dir.glob('*.md'))
    print(f'📁 Found {len(doc_files)} document files in healthcare_docs/')
    
    if len(doc_files) > stats['documents'] or stats['documents'] == 0:
        print('⬆️ Loading documents...')
        from scripts.load_healthcare_docs import load_all_documents
        load_all_documents(docs_dir)
        stats = chroma.get_collection_stats()
        print(f'After loading - Documents: {stats[\"documents\"]}')
    else:
        print('✅ Documents already up to date')
else:
    print(f'⚠️ Healthcare docs directory not found: {docs_dir}')
"

# ─── 5. FastAPI 서버 실행 ───
echo "🌐 Starting FastAPI server on port ${PORT:-8000}..."
echo "   Model: ${MODEL_NAME}"
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
