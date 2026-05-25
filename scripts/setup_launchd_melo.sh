#!/bin/bash
# =====================================================
# MeloTTS 사이드카 launchd 자동 시작 등록 (Mac, 호스트 네이티브)
# 로그인 시 .venv-tts 사이드카(8181)를 자동 기동하고, 죽으면 재시작(KeepAlive).
# =====================================================
# 사용법:
#   ./scripts/setup_launchd_melo.sh          # 등록 + 즉시 시작
#   ./scripts/setup_launchd_melo.sh remove   # 제거
#
# 전제: ./scripts/setup_melo_tts.sh 로 .venv-tts 구성 완료.
# =====================================================

set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_NAME="com.healthcare-rag.melo-sidecar"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"

remove_service() {
    echo -e "${YELLOW}MeloTTS 사이드카 서비스 제거 중...${NC}"
    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    rm -f "$PLIST_PATH"
    echo -e "${GREEN}✅ 제거 완료${NC}"
}

install_service() {
    echo -e "${GREEN}🔧 MeloTTS 사이드카 launchd 등록${NC}"
    echo "  프로젝트: ${PROJECT_DIR}"

    if [ ! -x "$PROJECT_DIR/.venv-tts/bin/python" ]; then
        echo -e "${YELLOW}⚠️  .venv-tts 가 없습니다. 먼저 ./scripts/setup_melo_tts.sh 를 실행하세요.${NC}"
        exit 1
    fi

    launchctl unload "$PLIST_PATH" 2>/dev/null || true
    mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/logs"

    cat > "$PLIST_PATH" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${PROJECT_DIR}/scripts/run_melo_sidecar.sh</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>${PROJECT_DIR}/logs/melo_sidecar_stdout.log</string>

    <key>StandardErrorPath</key>
    <string>${PROJECT_DIR}/logs/melo_sidecar_stderr.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>HOME</key>
        <string>${HOME}</string>
        <key>MELO_PORT</key>
        <string>8181</string>
    </dict>

    <key>ThrottleInterval</key>
    <integer>10</integer>
</dict>
</plist>
PLIST

    launchctl load "$PLIST_PATH"
    echo -e "${GREEN}✅ 등록 완료 — 로그인 시 자동 기동, 8181 포트${NC}"
    echo "  plist: $PLIST_PATH"
    echo "  로그 : $PROJECT_DIR/logs/melo_sidecar_stdout.log"
    echo "  상태 : curl -s http://127.0.0.1:8181/health"
}

if [ "$1" = "remove" ]; then
    remove_service
else
    install_service
fi
