#!/bin/bash
# =====================================================
# launchd 서비스 등록 스크립트
# 부팅 시 자동으로 챗봇 서버 시작
# =====================================================
# 사용법:
#   ./scripts/setup_launchd.sh          # 서비스 등록
#   ./scripts/setup_launchd.sh remove   # 서비스 제거
# =====================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_NAME="com.healthcare-rag.server"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"

# .env.local 또는 .env에서 변수 로드
ENV_FILE="$PROJECT_DIR/.env.local"
[ ! -f "$ENV_FILE" ] && ENV_FILE="$PROJECT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    eval "$(grep -E '^(OLLAMA_MODEL|API_PORT)=' "$ENV_FILE" | head -2)"
fi

OLLAMA_MODEL="${OLLAMA_MODEL:-kanana-counseling}"
API_PORT="${API_PORT:-8000}"

remove_service() {
    echo -e "${YELLOW}서비스 제거 중...${NC}"
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    rm -f "$PLIST_PATH"
    echo -e "${GREEN}✅ 서비스 제거 완료${NC}"
}

install_service() {
    echo -e "${GREEN}🔧 launchd 서비스 등록${NC}"
    echo "  프로젝트: ${PROJECT_DIR}"
    echo ""

    # 기존 서비스 중지
    launchctl unload "$PLIST_PATH" 2>/dev/null || true

    mkdir -p "$HOME/Library/LaunchAgents"

    cat > "$PLIST_PATH" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${PROJECT_DIR}/deploy_local.sh</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <false/>

    <key>StandardOutPath</key>
    <string>${PROJECT_DIR}/logs/launchd_stdout.log</string>

    <key>StandardErrorPath</key>
    <string>${PROJECT_DIR}/logs/launchd_stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>HOME</key>
        <string>${HOME}</string>
        <key>OLLAMA_MODEL</key>
        <string>${OLLAMA_MODEL}</string>
    </dict>

    <!-- 부팅 후 30초 대기 (네트워크 준비) -->
    <key>ThrottleInterval</key>
    <integer>30</integer>
</dict>
</plist>
PLIST

    # 서비스 로드
    launchctl load "$PLIST_PATH"

    echo -e "${GREEN}✅ launchd 서비스 등록 완료${NC}"
    echo ""
    echo "  plist 경로: $PLIST_PATH"
    echo ""
    echo "  관리 명령어:"
    echo "    상태:  launchctl list | grep healthcare"
    echo "    시작:  launchctl start ${PLIST_NAME}"
    echo "    중지:  launchctl stop ${PLIST_NAME}"
    echo "    제거:  $0 remove"
    echo ""
    echo "  로그:"
    echo "    tail -f ${PROJECT_DIR}/logs/launchd_stdout.log"
    echo "    tail -f ${PROJECT_DIR}/logs/launchd_stderr.log"
}

case "${1:-}" in
    remove|uninstall)
        remove_service
        ;;
    *)
        install_service
        ;;
esac
