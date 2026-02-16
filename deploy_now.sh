#!/bin/bash
# Cloud Run 배포만 실행 (빌드/푸시 완료 상태)
PROJECT_ID="rag-healthcare-483412"
REGION="asia-northeast3"
SERVICE_NAME="healthcare-rag-chatbot"
OLLAMA_MODEL="kanana-base"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "Cloud Run 배포 시작..."
echo "  이미지: ${IMAGE_NAME}:latest"
echo "  모델: ${OLLAMA_MODEL}"

gcloud run deploy ${SERVICE_NAME} \
    --quiet \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8080 \
    --memory 8Gi \
    --cpu 4 \
    --min-instances 0 \
    --max-instances 2 \
    --timeout 300 \
    --cpu-boost \
    --no-cpu-throttling \
    --concurrency 5 \
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

echo ""
echo "배포 완료!"
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed --region ${REGION} --project ${PROJECT_ID} \
    --format 'value(status.url)')
echo "URL: ${SERVICE_URL}"
echo "테스트: curl ${SERVICE_URL}/health"
