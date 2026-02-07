#!/bin/bash
# Cloud Run 배포 스크립트 (범용 모델 지원)
# .env의 OLLAMA_MODEL에 지정된 모델로 자동 배포
#
# 사전 준비:
#   1. models/{OLLAMA_MODEL}.gguf 파일 배치
#   2. models/Modelfile.{OLLAMA_MODEL} 템플릿 작성
#   3. .env에 OLLAMA_MODEL=모델명 설정

set -e

# .env 파일 로드
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -E '^GCP_|^SERVICE_NAME=|^OLLAMA_MODEL=' | xargs)
fi

# 설정
PROJECT_ID="${GCP_PROJECT_ID:-rag-healthcare-483412}"
REGION="${GCP_REGION:-asia-northeast3}"
SERVICE_NAME="${SERVICE_NAME:-healthcare-rag-chatbot}"
OLLAMA_MODEL="${OLLAMA_MODEL:-kanana-counseling}"
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
echo "  모델: ${OLLAMA_MODEL}"

# 1. 로컬에서 Docker 빌드 (캐시 활용)
echo -e "\n${YELLOW}📦 로컬에서 Docker 이미지 빌드 중... (캐시 활용)${NC}"
docker build -t ${IMAGE_NAME}:latest -f Dockerfile.ollama .

# 2. GCR에 푸시
echo -e "\n${YELLOW}📤 GCR에 이미지 푸시 중...${NC}"
docker push ${IMAGE_NAME}:latest

# 3. Cloud Run 배포
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
    --add-cloudsql-instances ${PROJECT_ID}:${REGION}:healthcare-db \
    --set-env-vars "CHROMA_IN_MEMORY=false" \
    --set-env-vars "CHROMA_PERSIST_DIR=/app/data/chroma" \
    --set-env-vars "OLLAMA_MODEL=${OLLAMA_MODEL}" \
    --set-env-vars "OLLAMA_BASE_URL=http://localhost:11434" \
    --set-env-vars "USE_LANGCHAIN_STORE=true" \
    --set-env-vars "GRAPHRAG_ENABLED=true" \
    --set-env-vars "DB_HOST=/cloudsql/${PROJECT_ID}:${REGION}:healthcare-db" \
    --set-env-vars "DB_NAME=healthcare" \
    --set-env-vars "DB_USER=postgres" \
    --set-secrets "DB_PASSWORD=db-password:latest" \
    --project ${PROJECT_ID}

# 4. 로컬 이미지 삭제 (디스크 절약)
echo -e "\n${YELLOW}🗑️ 로컬 이미지 삭제 중...${NC}"
docker rmi ${IMAGE_NAME}:latest 2>/dev/null || true

# 5. 서비스 URL 확인
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
