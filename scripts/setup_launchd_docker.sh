#!/bin/bash
# =====================================================
# launchd 서비스 등록 스크립트 (Docker 버전)
# 재부팅 시 Docker 컨테이너 자동 기동
# =====================================================
# 사용법:
#   ./scripts/setup_launchd_docker.sh          # 서비스 등록
#   ./scripts/setup_launchd_docker.sh remove   # 서비스 제거
#
# 전제 조건:
#   - Docker Desktop이 로그인 항목에 등록되어 있어야 함
#   - 시스템 설정 → 일반 → 로그인 항목 → Docker 추가
# =====================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_NAME="com.healthcare-rag.docker"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"

remove_service() {
    echo -e "${YELLOW}서비스 제거 중...${NC}"
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    rm -f "$PLIST_PATH"
    echo -e "${GREEN}✅ 서비스 제거 완료${NC}"
}

install_service() {
    echo -e "${GREEN}🔧 launchd Docker 서비스 등록${NC}"
    echo "  프로젝트: ${PROJECT_DIR}"
    echo ""

    # 기존 서비스 중지
    launchctl unload "$PLIST_PATH" 2>/dev/null || true

    mkdir -p "$HOME/Library/LaunchAgents"
    mkdir -p "$PROJECT_DIR/logs"

    cat > "$PLIST_PATH" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${PROJECT_DIR}/deploy_docker.sh</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <false/>

    <key>StandardOutPath</key>
    <string>${PROJECT_DIR}/logs/launchd_docker_stdout.log</string>

    <key>StandardErrorPath</key>
    <string>${PROJECT_DIR}/logs/launchd_docker_stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>HOME</key>
        <string>${HOME}</string>
    </dict>

    <!-- Docker Desktop 시작 대기 (재부팅 후 60초) -->
    <key>ThrottleInterval</key>
    <integer>60</integer>
</dict>
</plist>
PLIST

    launchctl load "$PLIST_PATH"

    echo -e "${GREEN}✅ launchd Docker 서비스 등록 완료${NC}"
    echo ""
    echo "  plist 경로: $PLIST_PATH"
    echo ""
    echo "  ⚠️  Docker Desktop도 로그인 항목에 추가해야 합니다:"
    echo "     시스템 설정 → 일반 → 로그인 항목 → Docker 추가"
    echo ""
    echo "  관리 명령어:"
    echo "    상태:  launchctl list | grep healthcare-rag.docker"
    echo "    시작:  launchctl start ${PLIST_NAME}"
    echo "    중지:  launchctl stop ${PLIST_NAME}"
    echo "    제거:  $0 remove"
    echo ""
    echo "  로그:"
    echo "    tail -f ${PROJECT_DIR}/logs/launchd_docker_stdout.log"
}

case "${1:-}" in
    remove|uninstall)
        remove_service
        ;;
    *)
        install_service
        ;;
esac
