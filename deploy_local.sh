#!/bin/bash
# =====================================================
# 맥미니 로컬 배포 스크립트
# Cloud Run 대체 → Mac Mini M4 상시 서버
# =====================================================
# 사전 요구사항:
#   1. Ollama 설치 (brew install ollama)
#   2. kanana-counseling 모델 등록 (ollama list로 확인)
#   3. .env.local 파일 설정 (Cloud SQL 접속 정보)
#   4. Cloud SQL Auth Proxy 설치 (선택: 보안 연결)
#   5. ngrok 설치 + authtoken 등록 (brew install ngrok)
#
# 사용법:
#   ./deploy_local.sh              # 전체 시작
#   ./deploy_local.sh --stop       # 전체 중지
#   ./deploy_local.sh --status     # 상태 확인
#   ./deploy_local.sh --restart    # 재시작
# =====================================================

set -e

# 색상 출력
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PID_DIR="$SCRIPT_DIR/.pids"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$PID_DIR" "$LOG_DIR"

# .env.local 로드 (.env.local 우선, 없으면 .env)
ENV_FILE=".env.local"
if [ ! -f "$ENV_FILE" ]; then
    ENV_FILE=".env"
fi
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | grep -v '^\s*$' | xargs)
fi

OLLAMA_MODEL="${OLLAMA_MODEL:-kanana-counseling}"
API_PORT="${API_PORT:-8000}"

# ── 함수 정의 ──

status_check() {
    echo -e "${BLUE}═══ 서비스 상태 ═══${NC}"

    # Ollama
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        MODEL_COUNT=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('models',[])))" 2>/dev/null || echo "?")
        echo -e "  Ollama:    ${GREEN}● 실행 중${NC} (모델 ${MODEL_COUNT}개)"
    else
        echo -e "  Ollama:    ${RED}○ 중지${NC}"
    fi

    # FastAPI
    if curl -s "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
        DOC_COUNT=$(curl -s "http://localhost:${API_PORT}/health" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('stats',{}).get('documents','?'))" 2>/dev/null || echo "?")
        echo -e "  FastAPI:   ${GREEN}● 실행 중${NC} (포트 ${API_PORT}, 문서 ${DOC_COUNT}개)"
    else
        echo -e "  FastAPI:   ${RED}○ 중지${NC}"
    fi

    # ngrok Tunnel
    if pgrep -f "ngrok" > /dev/null 2>&1; then
        NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 -c "import sys,json;t=json.load(sys.stdin).get('tunnels',[]);print(t[0]['public_url'] if t else '?')" 2>/dev/null || echo "?")
        echo -e "  ngrok:     ${GREEN}● 실행 중${NC} (${NGROK_URL})"
    else
        echo -e "  ngrok:     ${RED}○ 중지${NC} (HuggingFace 연결 불가)"
    fi

    # Cloud SQL Proxy
    if pgrep -f "cloud-sql-proxy\|cloud_sql_proxy" > /dev/null 2>&1; then
        echo -e "  SQL Proxy: ${GREEN}● 실행 중${NC}"
    else
        echo -e "  SQL Proxy: ${YELLOW}○ 미사용${NC} (직접 TCP 연결)"
    fi

    echo ""
}

stop_all() {
    echo -e "${YELLOW}🛑 서비스 중지 중...${NC}"

    # FastAPI
    if [ -f "$PID_DIR/fastapi.pid" ]; then
        kill "$(cat "$PID_DIR/fastapi.pid")" 2>/dev/null && echo "  FastAPI 중지됨" || true
        rm -f "$PID_DIR/fastapi.pid"
    fi
    # uvicorn 프로세스 정리
    pkill -f "uvicorn app.main:app" 2>/dev/null || true

    # ngrok
    if [ -f "$PID_DIR/ngrok.pid" ]; then
        kill "$(cat "$PID_DIR/ngrok.pid")" 2>/dev/null && echo "  ngrok 중지됨" || true
        rm -f "$PID_DIR/ngrok.pid"
    fi
    pkill -f "ngrok" 2>/dev/null || true

    # Cloud SQL Proxy
    if [ -f "$PID_DIR/sqlproxy.pid" ]; then
        kill "$(cat "$PID_DIR/sqlproxy.pid")" 2>/dev/null && echo "  SQL Proxy 중지됨" || true
        rm -f "$PID_DIR/sqlproxy.pid"
    fi

    echo -e "${GREEN}✅ 모든 서비스 중지 완료${NC}"
}

start_all() {
    echo -e "${GREEN}🚀 맥미니 로컬 서버 시작${NC}"
    echo "  환경: Mac Mini M4 ($(sysctl -n hw.ncpu) cores, $(sysctl -n hw.memsize | awk '{printf "%.0f", $0/1073741824}')GB RAM)"
    echo "  모델: ${OLLAMA_MODEL}"
    echo "  포트: ${API_PORT}"
    echo ""

    # ── 1. Ollama 확인 ──
    echo -e "${YELLOW}[1/4] Ollama 확인...${NC}"
    if ! command -v ollama &> /dev/null; then
        echo -e "${RED}❌ Ollama가 설치되지 않았습니다. brew install ollama${NC}"
        exit 1
    fi

    # Ollama 서버가 안 돌고 있으면 시작
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "  Ollama 서버 시작 중..."
        ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
        for i in $(seq 1 15); do
            if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                break
            fi
            sleep 1
        done
    fi

    # 모델 확인
    if ollama list 2>/dev/null | grep -q "${OLLAMA_MODEL}"; then
        echo -e "  ${GREEN}✅ ${OLLAMA_MODEL} 모델 준비됨${NC}"
    else
        echo -e "  ${RED}❌ ${OLLAMA_MODEL} 모델이 없습니다.${NC}"
        echo "  ollama create ${OLLAMA_MODEL} -f models/Modelfile.${OLLAMA_MODEL}"
        exit 1
    fi

    # 워밍업 (Metal GPU 활용)
    echo "  모델 워밍업 중 (Metal GPU)..."
    curl -s http://localhost:11434/api/generate \
        -d "{\"model\": \"${OLLAMA_MODEL}\", \"prompt\": \"안녕\", \"stream\": false}" > /dev/null 2>&1 || true
    echo -e "  ${GREEN}✅ 워밍업 완료${NC}"

    # ── 2. Cloud SQL Proxy (선택) ──
    echo -e "\n${YELLOW}[2/4] 데이터베이스 연결...${NC}"
    if [ -n "$CLOUD_SQL_INSTANCE" ] && command -v cloud-sql-proxy &> /dev/null; then
        echo "  Cloud SQL Auth Proxy 시작..."
        cloud-sql-proxy "$CLOUD_SQL_INSTANCE" \
            --port "${DB_PROXY_PORT:-5432}" \
            > "$LOG_DIR/sqlproxy.log" 2>&1 &
        echo $! > "$PID_DIR/sqlproxy.pid"
        sleep 2
        echo -e "  ${GREEN}✅ Cloud SQL Proxy 시작됨 (localhost:${DB_PROXY_PORT:-5432})${NC}"
    elif [ -n "$DATABASE_URL" ]; then
        echo -e "  ${GREEN}✅ 직접 TCP 연결 사용${NC}"
    else
        echo -e "  ${YELLOW}⚠️ DB 미설정 (로컬 ChromaDB 사용)${NC}"
    fi

    # ── 3. FastAPI 서버 ──
    echo -e "\n${YELLOW}[3/4] FastAPI 서버 시작...${NC}"

    # 가상환경 활성화
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    fi

    # 기존 프로세스 정리
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
    sleep 1

    # 서버 시작 (백그라운드)
    nohup uvicorn app.main:app \
        --host 0.0.0.0 \
        --port "$API_PORT" \
        > "$LOG_DIR/fastapi.log" 2>&1 &
    echo $! > "$PID_DIR/fastapi.pid"

    # 서버 준비 대기
    echo "  서버 시작 대기 중..."
    for i in $(seq 1 30); do
        if curl -s "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
            echo -e "  ${GREEN}✅ FastAPI 서버 시작 완료 (${i}초)${NC}"
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo -e "  ${YELLOW}⚠️ 서버 시작 중... (로그: logs/fastapi.log)${NC}"
        fi
        sleep 1
    done

    # ── 4. ngrok Tunnel ──
    echo -e "\n${YELLOW}[4/4] ngrok Tunnel...${NC}"
    NGROK_DOMAIN="${NGROK_DOMAIN:-acronymous-nonobsessive-chong.ngrok-free.dev}"
    if command -v ngrok &> /dev/null; then
        pkill -f "ngrok" 2>/dev/null || true
        sleep 1
        nohup ngrok http "$API_PORT" --domain "$NGROK_DOMAIN" \
            > "$LOG_DIR/ngrok.log" 2>&1 &
        echo $! > "$PID_DIR/ngrok.pid"
        sleep 3
        echo -e "  ${GREEN}✅ ngrok 시작됨${NC}"
        echo -e "  🔗 공개 URL: https://${NGROK_DOMAIN}"
    else
        echo -e "  ${YELLOW}⚠️ ngrok 미설치 (brew install ngrok)${NC}"
    fi

    # ── 완료 ──
    echo -e "\n${GREEN}═══════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ 맥미니 로컬 서버 배포 완료!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════${NC}"
    echo ""
    echo "  로컬 API:  http://localhost:${API_PORT}"
    echo "  공개 URL:  https://${NGROK_DOMAIN:-acronymous-nonobsessive-chong.ngrok-free.dev}"
    echo "  Swagger:   http://localhost:${API_PORT}/docs"
    echo "  헬스체크:  http://localhost:${API_PORT}/health"
    echo ""
    echo "  로그 확인: tail -f logs/fastapi.log"
    echo "  상태 확인: ./deploy_local.sh --status"
    echo "  서비스 중지: ./deploy_local.sh --stop"
    echo ""
}

# ── 명령어 분기 ──
case "${1:-}" in
    --stop)
        stop_all
        ;;
    --status)
        status_check
        ;;
    --restart)
        stop_all
        sleep 2
        start_all
        ;;
    *)
        start_all
        ;;
esac
