#!/bin/bash
# EXAONE GGUF 모델을 Ollama에 등록하는 스크립트
# 사용법: ./scripts/setup_exaone.sh [1.2b|2.4b]

set -e

MODEL_SIZE="${1:-1.2b}"
MODELS_DIR="$(pwd)/models"

mkdir -p "$MODELS_DIR"

echo "🤖 EXAONE 모델 설정 스크립트"
echo "   선택된 모델: EXAONE-${MODEL_SIZE^^}"

# HuggingFace에서 GGUF 다운로드
if [ "$MODEL_SIZE" = "1.2b" ]; then
    MODEL_NAME="exaone1.2b"
    HF_REPO="LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF"
    # Q4_K_M 양자화 버전 사용 (CPU 최적화, 더 빠름)
    GGUF_FILE="EXAONE-4.0-1.2B-Q4_K_M.gguf"
    DOWNLOAD_URL="https://huggingface.co/${HF_REPO}/resolve/main/${GGUF_FILE}"
elif [ "$MODEL_SIZE" = "2.4b" ]; then
    MODEL_NAME="exaone2.4b"
    HF_REPO="LGAI-EXAONE/EXAONE-Deep-2.4B-GGUF"
    # Q4_K_M 양자화 버전 사용 (CPU 최적화, 더 빠름)
    GGUF_FILE="EXAONE-Deep-2.4B-Q4_K_M.gguf"
    DOWNLOAD_URL="https://huggingface.co/${HF_REPO}/resolve/main/${GGUF_FILE}"
else
    echo "❌ 지원하지 않는 모델 크기: $MODEL_SIZE"
    echo "   사용법: ./scripts/setup_exaone.sh [1.2b|2.4b]"
    exit 1
fi

GGUF_PATH="${MODELS_DIR}/${GGUF_FILE}"

# GGUF 파일 다운로드 (없는 경우)
if [ ! -f "$GGUF_PATH" ]; then
    echo "⬇️ GGUF 파일 다운로드 중..."
    echo "   URL: $DOWNLOAD_URL"
    
    # huggingface-cli 사용 시도, 없으면 curl 사용
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$HF_REPO" "$GGUF_FILE" --local-dir "$MODELS_DIR"
    else
        curl -L -o "$GGUF_PATH" "$DOWNLOAD_URL"
    fi
    
    echo "✅ 다운로드 완료: $GGUF_PATH"
else
    echo "✅ GGUF 파일 이미 존재: $GGUF_PATH"
fi

# Modelfile 생성 (절대 경로 사용)
MODELFILE_PATH="${MODELS_DIR}/Modelfile.${MODEL_NAME}"

cat > "$MODELFILE_PATH" << EOF
# EXAONE ${MODEL_SIZE^^} 모델 설정
FROM ${GGUF_PATH}

# 한국어 헬스케어 챗봇용 파라미터
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 1024

# 시스템 프롬프트 (한국어 강제)
SYSTEM """당신은 치매노인을 돌보는 따뜻하고 친절한 AI 도우미입니다.
반드시 한국어로만 응답하세요. 한자를 사용하지 마세요."""
EOF

echo "📝 Modelfile 생성: $MODELFILE_PATH"
echo "   FROM 경로: $GGUF_PATH"

# Ollama에 모델 등록
echo "🔧 Ollama에 모델 등록 중..."
ollama create "$MODEL_NAME" -f "$MODELFILE_PATH"

echo ""
echo "✅ 설정 완료!"
echo ""
echo "사용 방법:"
echo "  1. 환경변수로 모델 변경:"
echo "     export OLLAMA_MODEL=${MODEL_NAME}"
echo ""
echo "  2. .env 파일에 추가:"
echo "     OLLAMA_MODEL=${MODEL_NAME}"
echo ""
echo "  3. 직접 테스트:"
echo "     ollama run ${MODEL_NAME} \"안녕하세요\""
echo ""
echo "  4. 서버 시작:"
echo "     OLLAMA_MODEL=${MODEL_NAME} ./server.sh"
