#!/bin/bash
# Cloud Run용 시작 스크립트: Ollama + Qwen + FastAPI

set -e

# UTF-8 환경 및 Ollama 로그 레벨 설정
export PYTHONIOENCODING=utf-8
export OLLAMA_DEBUG=0

echo "🚀 Starting Ollama server..."
ollama serve 2>&1 | grep -v "print_info\|llama_\|ggml_\|rope_\|vocab\|token" &
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
# kanana-counseling = finetuned 모델 (로컬 GGUF 파일 사용)
# kanana = kakaocorp/kanana-nano-2.1b-instruct (HuggingFace GGUF: ch00n/kanana-nano-2.1b-instruct-Q4_K_M-GGUF)
MODEL_NAME="${OLLAMA_MODEL:-kanana-counseling}"
echo "📦 Checking model: ${MODEL_NAME}..."

if ! ollama list | grep -q "${MODEL_NAME}"; then
    # kanana-counseling (finetuned 모델) - 로컬 GGUF 파일 사용
    if [ "${MODEL_NAME}" = "kanana-counseling" ]; then
        echo "📝 Registering finetuned model: ${MODEL_NAME}..."
        GGUF_PATH="/app/models/kanana-counseling-q4_k_m.gguf"
        
        if [ -f "${GGUF_PATH}" ]; then
            # Modelfile 생성 및 등록
            cat > /tmp/Modelfile.${MODEL_NAME} << 'EOF'
FROM /app/models/kanana-counseling-q4_k_m.gguf

SYSTEM """당신은 노인건강전문상담사입니다.
- 2~3문장으로 간결하게 답변하세요
- 공감 후 질문으로 문제를 파악하세요
- 일상에서 실천할 수 있는 건강 습관을 안내하세요
- 심각한 경우에만 병원 진료를 권유하세요

[금지사항]
절대로 "추가로 궁금한 점이나 불편한 점이 있으시면 언제든지 말씀해 주세요"라고 말하지 마세요.
마무리 인사 없이 핵심 내용만 전달하세요."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 256
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "추가로 궁금한"
PARAMETER stop "궁금한 점이나"
PARAMETER stop "불편한 점이"
PARAMETER stop "언제든지 말씀"

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
EOF
            ollama create ${MODEL_NAME} -f /tmp/Modelfile.${MODEL_NAME}
            echo "✅ ${MODEL_NAME} model registered!"
        else
            echo "❌ GGUF file not found: ${GGUF_PATH}"
            echo "   Falling back to base kanana model..."
            MODEL_NAME="kanana"
        fi
    fi
    
    # kanana 모델은 HuggingFace에서 GGUF 다운로드 후 등록
    if [ "${MODEL_NAME}" = "kanana" ]; then
        echo "⬇️ Downloading kanana-nano-2.1b-instruct from HuggingFace..."
        echo "   Source: ch00n/kanana-nano-2.1b-instruct-Q4_K_M-GGUF"
        GGUF_URL="https://huggingface.co/ch00n/kanana-nano-2.1b-instruct-Q4_K_M-GGUF/resolve/main/kanana-nano-2.1b-instruct-q4_k_m.gguf"
        GGUF_PATH="/app/models/${MODEL_NAME}.gguf"
        
        # 다운로드 (재시도 포함)
        for attempt in 1 2 3; do
            if curl -L --retry 3 --retry-delay 5 -o "${GGUF_PATH}" "${GGUF_URL}"; then
                echo "✅ GGUF downloaded successfully!"
                break
            else
                echo "⚠️ Download attempt $attempt failed, retrying..."
                sleep 5
            fi
            if [ $attempt -eq 3 ]; then
                echo "❌ Failed to download GGUF after 3 attempts"
                exit 1
            fi
        done
        
        # Modelfile 생성 및 등록
        cat > /tmp/Modelfile.${MODEL_NAME} << EOF
FROM ${GGUF_PATH}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_predict 512
SYSTEM "당신은 치매노인을 돌보는 따뜻하고 친절한 AI 도우미입니다. 반드시 한국어로만 응답하세요."
EOF
        echo "📝 Registering ${MODEL_NAME} model with Ollama..."
        ollama create ${MODEL_NAME} -f /tmp/Modelfile.${MODEL_NAME}
        echo "✅ ${MODEL_NAME} model registered!"
    elif [ "${MODEL_NAME}" != "kanana-counseling" ]; then
        # 일반 Ollama 모델 pull (kanana, kanana-counseling이 아닌 경우)
        echo "⬇️ Pulling model: ${MODEL_NAME} (this may take a while on first run)..."
        echo "   Progress logs suppressed. Please wait..."
        # 3번 재시도 (진행 상황 로그 숨김)
        for attempt in 1 2 3; do
            if ollama pull ${MODEL_NAME} 2>&1 | grep -E "(success|error|failed|pulling [a-f0-9]+:.*100%)" || [ ${PIPESTATUS[0]} -eq 0 ]; then
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
else
    echo "✅ Model already available!"
fi

# 모델 검증 (한글 테스트) - 로그 간소화
echo "🔍 Verifying model..."
KOREAN_TEST=$(curl -s http://localhost:11434/api/generate -d "{\"model\": \"${MODEL_NAME}\", \"prompt\": \"안녕하세요라고 말해주세요\", \"stream\": false}" 2>&1)
if echo "$KOREAN_TEST" | grep -q "안녕"; then
    echo "✅ Korean language support verified!"
else
    echo "⚠️ Model may have issues with Korean, but continuing..."
fi

# 모델 미리 로드 (워밍업) - 로그 간소화
echo "🔥 Warming up model..."
curl -s http://localhost:11434/api/generate -d "{\"model\": \"${MODEL_NAME}\", \"prompt\": \"hello\", \"stream\": false}" > /dev/null 2>&1 || true
echo "✅ Model ready!"

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
# 새 문서가 추가된 경우에도 자동으로 로드
echo "📄 Checking and loading documents..."
python3 -c "
from pathlib import Path
from app.vector_store import get_chroma_handler
chroma = get_chroma_handler()
stats = chroma.get_collection_stats()
print(f'Current - Documents: {stats[\"documents\"]}')
print(f'          Conversations: {stats[\"conversations\"]}')
print(f'          Profiles: {stats[\"patient_profiles\"]}')

# 항상 healthcare_docs 폴더의 문서를 확인하고 새 문서가 있으면 로드
import sys
sys.path.insert(0, '/app')
docs_dir = Path('/app/data/healthcare_docs')

if docs_dir.exists():
    # 폴더 내 문서 파일 수 확인
    doc_files = list(docs_dir.glob('*.txt')) + list(docs_dir.glob('*.md'))
    print(f'📁 Found {len(doc_files)} document files in healthcare_docs/')
    
    if len(doc_files) > stats['documents']:
        print('⬆️ New documents detected, reloading all documents...')
        from scripts.load_healthcare_docs import load_all_documents
        load_all_documents(docs_dir)
        stats = chroma.get_collection_stats()
        print(f'After loading - Documents: {stats[\"documents\"]}')
    elif stats['documents'] == 0:
        print('⚠️ No documents in DB, loading healthcare docs...')
        from scripts.load_healthcare_docs import load_all_documents
        load_all_documents(docs_dir)
        stats = chroma.get_collection_stats()
        print(f'After loading - Documents: {stats[\"documents\"]}')
    else:
        print('✅ Documents already up to date')
else:
    print(f'⚠️ Healthcare docs directory not found: {docs_dir}')
"

# FastAPI 앱 실행
echo "🌐 Starting FastAPI server on port ${PORT:-8000}..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
