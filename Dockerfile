# 치매노인 맞춤형 헬스케어 RAG 챗봇 - Cloud Run 배포용
# 환경변수 OLLAMA_MODEL로 모델을 지정하면 자동으로 해당 모델을 등록/사용
FROM python:3.12-slim

# Build argument
ARG OLLAMA_MODEL=k-exaone-counseling

# 환경변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LOG_LEVEL=INFO \
    LOG_TO_CONSOLE=true \
    LOG_TO_FILE=true \
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_MODELS=/app/ollama_models \
    OLLAMA_BASE_URL=http://localhost:11434 \
    OLLAMA_MODEL=${OLLAMA_MODEL} \
    PYTHONIOENCODING=utf-8

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 + Ollama
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY app/ ./app/
COPY data/ ./data/
COPY scripts/ ./scripts/

# 모델 디렉토리 복사 (GGUF + Modelfile 포함)
COPY models/ ./models/

# 로그 디렉토리 생성 및 스크립트 권한
RUN mkdir -p /app/logs /app/ollama_models && \
    chmod +x ./scripts/start_with_ollama.sh

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# 포트 노출
EXPOSE 8000

# 서버 실행 (Ollama + FastAPI)
CMD ["./scripts/start_with_ollama.sh"]
