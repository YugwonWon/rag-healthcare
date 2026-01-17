# 치매노인 맞춤형 헬스케어 RAG 챗봇 - Cloud Run 배포용 Dockerfile
FROM python:3.12-slim

# 환경변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LOG_LEVEL=INFO \
    LOG_TO_CONSOLE=true \
    LOG_TO_FILE=true \
    OLLAMA_HOST=http://localhost:11434

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

# GGUF 모델과 Modelfile 복사
COPY finetuning/output/kanana-counseling-merged/kanana-counseling-q4_k_m.gguf ./models/
COPY finetuning/output/kanana-counseling-merged/Modelfile.docker ./models/Modelfile

# 로그 디렉토리 생성
RUN mkdir -p /app/logs

# 시작 스크립트 생성
RUN echo '#!/bin/bash\n\
ollama serve &\n\
sleep 3\n\
cd /app/models && ollama create kanana-counseling -f Modelfile\n\
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}' > /app/start.sh \
    && chmod +x /app/start.sh

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# 포트 노출
EXPOSE 8000

# 서버 실행 (Ollama + FastAPI)
CMD ["/app/start.sh"]
