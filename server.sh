#!/bin/bash
# 로컬 개발 서버 실행 스크립트

set -e

# 색상 출력
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🏥 치매노인 맞춤형 헬스케어 RAG 챗봇 시작${NC}"

# 환경 확인
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️ .env 파일이 없습니다. .env.example에서 복사합니다...${NC}"
    cp .env.example .env 2>/dev/null || echo "OLLAMA_MODEL=qwen2.5:3b" > .env
fi

# 가상환경 활성화 (있는 경우)
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Ollama 확인
echo -e "\n${YELLOW}🔍 Ollama 서버 확인 중...${NC}"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama 서버 실행 중"
    
    # 모델 확인
    MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
    if curl -s http://localhost:11434/api/tags | grep -q "${MODEL}"; then
        echo "✅ ${MODEL} 모델 사용 가능"
    else
        echo -e "${YELLOW}⚠️ ${MODEL} 모델이 없습니다. 다운로드를 시작합니다...${NC}"
        ollama pull ${MODEL}
    fi
else
    echo -e "${YELLOW}⚠️ Ollama 서버가 실행되지 않았습니다.${NC}"
    echo "   ollama serve 명령으로 Ollama를 시작하세요."
    echo "   OpenAI API를 fallback으로 사용합니다."
fi

# 샘플 데이터 생성 (없는 경우)
if [ ! -f "data/conversations/train_chat.jsonl" ]; then
    echo -e "\n${YELLOW}📝 샘플 대화 데이터 생성 중...${NC}"
    python finetuning/prepare_dataset.py --create-sample
fi

# 서버 시작
echo -e "\n${GREEN}🚀 FastAPI 서버 시작...${NC}"
echo "   API 문서: http://localhost:8000/docs"
echo "   헬스체크: http://localhost:8000/health"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
