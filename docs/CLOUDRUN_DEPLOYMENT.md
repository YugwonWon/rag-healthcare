# Cloud Run 배포 가이드

## 📋 사전 준비

### 1. GCP 프로젝트 설정

```bash
# GCP CLI 설치 및 로그인
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 필요한 API 활성화
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 2. 환경 변수 설정

```bash
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=asia-northeast3
export SERVICE_NAME=healthcare-rag-chatbot
```

## 🚀 배포 방법

### 방법 1: 자동 배포 스크립트 사용

```bash
# 스크립트 실행 권한 부여
chmod +x deploy_cloudrun.sh

# 배포 실행
./deploy_cloudrun.sh
```

### 방법 2: 수동 배포

```bash
# 1. Docker 이미지 빌드
docker build -t gcr.io/${GCP_PROJECT_ID}/${SERVICE_NAME}:latest .

# 2. GCR에 푸시
docker push gcr.io/${GCP_PROJECT_ID}/${SERVICE_NAME}:latest

# 3. Cloud Run 배포
gcloud run deploy ${SERVICE_NAME} \
    --image gcr.io/${GCP_PROJECT_ID}/${SERVICE_NAME}:latest \
    --platform managed \
    --region ${GCP_REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --set-env-vars "CHROMA_IN_MEMORY=true"
```

## ⚙️ 환경 변수 설정

Cloud Run에서 필요한 환경 변수:

| 변수명 | 설명 | 예시 |
|--------|------|------|
| `CHROMA_IN_MEMORY` | 인메모리 모드 사용 | `true` |
| `OLLAMA_BASE_URL` | Ollama 서버 URL | `http://ollama-server:11434` |
| `OPENAI_API_KEY` | OpenAI API 키 (Fallback) | `sk-...` |

```bash
gcloud run services update ${SERVICE_NAME} \
    --set-env-vars "OPENAI_API_KEY=sk-your-key"
```

## 🔧 Ollama 서버 연동

Cloud Run에서 Ollama를 사용하려면:

### 옵션 1: Compute Engine에 Ollama 서버 배포

```bash
# GPU 인스턴스 생성
gcloud compute instances create ollama-server \
    --zone=asia-northeast3-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --boot-disk-size=100GB \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud

# Ollama 설치 (인스턴스 내에서)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5:3b
```

### 옵션 2: OpenAI Fallback 사용

Ollama 없이 OpenAI API만 사용:

```bash
gcloud run services update ${SERVICE_NAME} \
    --set-env-vars "OPENAI_API_KEY=sk-your-key"
```

## 📊 모니터링

### 로그 확인

```bash
gcloud run services logs read ${SERVICE_NAME} --region ${GCP_REGION}
```

### 헬스 체크

```bash
curl https://your-service-url.run.app/health
```

## 🔒 보안 설정

### 인증 활성화

```bash
gcloud run services update ${SERVICE_NAME} --no-allow-unauthenticated
```

### IAM 설정

```bash
# 특정 사용자에게 접근 권한 부여
gcloud run services add-iam-policy-binding ${SERVICE_NAME} \
    --member="user:email@example.com" \
    --role="roles/run.invoker"
```

## 💡 비용 최적화

- `--min-instances 0`: 유휴 시 인스턴스 0개로 축소
- `--max-instances 10`: 최대 인스턴스 제한
- `--memory 2Gi`: 메모리 최적화 (임베딩 모델 크기 고려)

## 🐛 트러블슈팅

### 콜드 스타트 느림

임베딩 모델 로딩 시간이 길 경우:

```bash
gcloud run services update ${SERVICE_NAME} --min-instances 1
```

### 메모리 부족

```bash
gcloud run services update ${SERVICE_NAME} --memory 4Gi
```

### 타임아웃 에러

```bash
gcloud run services update ${SERVICE_NAME} --timeout 300
```
