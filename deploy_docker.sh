#!/bin/bash
# =====================================================
# 맥미니 Docker 배포 스크립트 (Ollama 네이티브 유지)
# =====================================================
# 전제 조건:
#   1. Docker Desktop 실행 중
#   2. Ollama 호스트 실행 중 (ollama serve 또는 자동 실행)
#   3. .env.local에 NGROK_AUTHTOKEN=... 설정
#
# 사용법:
#   ./deploy_docker.sh              # 시작
#   ./deploy_docker.sh --stop       # 종료
#   ./deploy_docker.sh --status     # 상태 확인
#   ./deploy_docker.sh --restart    # 재시작
#   ./deploy_docker.sh --logs       # 앱 로그 스트리밍
#   ./deploy_docker.sh --build      # 이미지 재빌드 후 시작
# =====================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_FILE="docker-compose.mac.yml"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# .env.local 로드
ENV_FILE=".env.local"
[ ! -f "$ENV_FILE" ] && ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

check_prereqs() {
    local ok=true

    # Docker
    if ! docker info > /dev/null 2>&1; then
        echo -e "  ${RED}❌ Docker Desktop이 실행되지 않았습니다.${NC}"
        echo "     Docker Desktop을 시작한 후 다시 실행하세요."
        ok=false
    fi

    # Ollama
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "  ${YELLOW}⚠️  Ollama가 실행되지 않았습니다. 시작합니다...${NC}"
        ollama serve > /dev/null 2>&1 &
        for i in $(seq 1 15); do
            if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                echo -e "  ${GREEN}✅ Ollama 시작됨${NC}"
                break
            fi
            sleep 1
        done
    fi

    # ngrok authtoken
    if [ -z "$NGROK_AUTHTOKEN" ]; then
        echo -e "  ${RED}❌ NGROK_AUTHTOKEN이 설정되지 않았습니다.${NC}"
        echo "     .env.local에 다음 줄을 추가하세요:"
        echo "     NGROK_AUTHTOKEN=your_token_here"
        echo "     (토큰: https://dashboard.ngrok.com/get-started/your-authtoken)"
        ok=false
    fi

    [ "$ok" = true ]
}

status_check() {
    echo -e "${BLUE}═══ 서비스 상태 ═══${NC}"

    # Ollama (호스트)
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        MODEL_COUNT=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json;print(len(json.load(sys.stdin).get('models',[])))" 2>/dev/null || echo "?")
        echo -e "  Ollama (호스트): ${GREEN}● 실행 중${NC} (모델 ${MODEL_COUNT}개)"
    else
        echo -e "  Ollama (호스트): ${RED}○ 중지${NC}"
    fi

    # FastAPI
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        DOC_COUNT=$(curl -s http://localhost:8000/health | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('stats',{}).get('documents','?'))" 2>/dev/null || echo "?")
        echo -e "  FastAPI (Docker): ${GREEN}● 실행 중${NC} (문서 ${DOC_COUNT}개)"
    else
        echo -e "  FastAPI (Docker): ${RED}○ 중지${NC}"
    fi

    # ngrok
    if curl -s http://localhost:4040/api/tunnels > /dev/null 2>&1; then
        NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | python3 -c "import sys,json;t=json.load(sys.stdin).get('tunnels',[]);print(t[0]['public_url'] if t else '?')" 2>/dev/null || echo "?")
        echo -e "  ngrok (Docker):  ${GREEN}● 실행 중${NC} (${NGROK_URL})"
    else
        echo -e "  ngrok (Docker):  ${RED}○ 중지${NC}"
    fi

    echo ""
}

start_all() {
    echo -e "${GREEN}🚀 맥미니 Docker 서버 시작${NC}"
    echo ""

    check_prereqs || exit 1

    echo -e "${YELLOW}[1/2] Docker 컨테이너 시작 중...${NC}"
    docker compose -f "$COMPOSE_FILE" up -d

    echo -e "\n${YELLOW}[2/2] 서버 준비 대기 중...${NC}"
    for i in $(seq 1 40); do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "  ${GREEN}✅ FastAPI 준비 완료 (${i}× 30초)${NC}"
            break
        fi
        if [ "$i" -eq 40 ]; then
            echo -e "  ${YELLOW}⚠️ 아직 시작 중... (로그: ./deploy_docker.sh --logs)${NC}"
        fi
        sleep 3
    done

    echo ""
    echo -e "${GREEN}═══════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ Docker 배포 완료!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════${NC}"
    echo ""
    echo "  로컬 API:   http://localhost:8000"
    echo "  공개 URL:   https://${NGROK_DOMAIN:-acronymous-nonobsessive-chong.ngrok-free.dev}"
    echo "  Swagger:    http://localhost:8000/docs"
    echo "  ngrok 대시: http://localhost:4040"
    echo ""
    echo "  로그: ./deploy_docker.sh --logs"
    echo "  상태: ./deploy_docker.sh --status"
    echo "  중지: ./deploy_docker.sh --stop"
    echo ""
}

stop_all() {
    echo -e "${YELLOW}🛑 Docker 컨테이너 중지 중...${NC}"
    docker compose -f "$COMPOSE_FILE" down
    echo -e "${GREEN}✅ 중지 완료${NC}"
}

case "${1:-}" in
    --stop|stop)
        stop_all
        ;;
    --status|status)
        status_check
        ;;
    --restart|restart)
        stop_all
        sleep 2
        start_all
        ;;
    --logs|logs)
        docker compose -f "$COMPOSE_FILE" logs -f app
        ;;
    --build|build)
        echo -e "${YELLOW}📦 이미지 재빌드 중...${NC}"
        docker compose -f "$COMPOSE_FILE" build --no-cache
        start_all
        ;;
    *)
        start_all
        ;;
esac
