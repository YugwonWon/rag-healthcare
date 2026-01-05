#!/bin/bash
# Cloud Run 배포 스크립트 (Cloud Build 사용 - 로컬 이미지 저장 없음)

set -e

# .env 파일 로드
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -E '^GCP_|^SERVICE_NAME=' | xargs)
fi

# 설정
PROJECT_ID="${GCP_PROJECT_ID:-rag-healthcare-483412}"
REGION="${GCP_REGION:-asia-northeast3}"
SERVICE_NAME="${SERVICE_NAME:-healthcare-rag-chatbot}"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# 색상 출력
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Cloud Run 배포 시작${NC}"
echo "  프로젝트: ${PROJECT_ID}"
echo "  리전: ${REGION}"
echo "  서비스: ${SERVICE_NAME}"

# 1. Cloud Build로 이미지 빌드 + GCR 푸시 (로컬 저장 없음)
# Ollama 포함 Dockerfile 사용
echo -e "\n${YELLOW}📦 Cloud Build로 이미지 빌드 중... (Ollama 포함, 로컬 저장 없음)${NC}"
gcloud builds submit \
    --config=cloudbuild.yaml \
    --project ${PROJECT_ID}

# 3. Cloud Run 배포
# LLM 내부 실행 (Ollama + Qwen2.5:3b)
echo -e "\n${YELLOW}🌐 Cloud Run에 배포 중...${NC}"
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 8Gi \
    --cpu 4 \
    --min-instances 0 \
    --max-instances 2 \
    --timeout 300 \
    --concurrency 5 \
    --cpu-boost \
    --execution-environment gen2 \
    --set-env-vars "CHROMA_IN_MEMORY=false" \
    --set-env-vars "CHROMA_PERSIST_DIR=/app/data/chroma" \
    --set-env-vars "OLLAMA_MODEL=qwen2.5:3b" \
    --set-env-vars "OLLAMA_BASE_URL=http://localhost:11434" \
    --project ${PROJECT_ID}

# 4. 서비스 URL 확인
echo -e "\n${GREEN}✅ 배포 완료!${NC}"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --project ${PROJECT_ID} \
    --format 'value(status.url)')

echo -e "🔗 서비스 URL: ${SERVICE_URL}"
echo -e "\n테스트 명령어:"
echo "  curl ${SERVICE_URL}/health"
echo "  curl -X POST ${SERVICE_URL}/chat -H 'Content-Type: application/json' -d '{\"nickname\":\"테스트\",\"message\":\"안녕하세요\"}'"
