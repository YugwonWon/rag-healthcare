#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# 데모 워치독 — 공개 URL(ngrok) 헬스를 주기 점검하고, 끊기면 체인을 따라 자동 복구.
#   Docker 데몬 크래시 → Docker Desktop 재시작
#   백엔드(app) 다운/행 → compose up -d
#   백엔드 정상인데 공개 URL만 실패 → ngrok 컨테이너 재시작 (터널 재수립)
#   STT/TTS 사이드카 다운 → launchd kickstart
# launchd(com.healthcare-rag.watchdog)에서 60초 간격으로 실행. 정상이면 즉시 종료.
# ──────────────────────────────────────────────────────────────────────────
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"

PUBLIC_URL="https://yugwon-macmini.tail7f37ba.ts.net/health"   # Tailscale Funnel
LOCAL_URL="http://localhost:8000/health"
COMPOSE_FILE="/Users/yugwon/Projects/rag-healthcare/docker-compose.mac.yml"
LOG="/Users/yugwon/Projects/rag-healthcare/logs/watchdog.log"
DOCKER="/usr/local/bin/docker"
COMPOSE="/usr/local/bin/docker-compose"   # 하이픈 바이너리 (TCC 환경에서 더 안전)
TS="/usr/local/bin/tailscale"
UID_NUM="$(id -u)"

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

# 공개 URL 200 이면 모든 게 정상 — 빠르게 종료 (대부분의 호출이 여기서 끝남)
code=$(curl -s -m 8 -H "ngrok-skip-browser-warning: true" -o /dev/null -w '%{http_code}' "$PUBLIC_URL" 2>/dev/null)
[ "$code" = "200" ] && exit 0

log "⚠️ 공개 URL 비정상(code=$code) — 복구 점검 시작"

# 1) Docker 데몬 살아있나
if ! $DOCKER info >/dev/null 2>&1; then
  log "  Docker 데몬 down → Docker Desktop 시작"
  open -a Docker
  for i in $(seq 1 45); do $DOCKER info >/dev/null 2>&1 && { log "  Docker 데몬 복구됨(${i}x2s)"; break; }; sleep 2; done
fi

# 2) 로컬 백엔드(8000) 점검 → 컨테이너 복구
if ! curl -s -m 5 -o /dev/null "$LOCAL_URL" 2>/dev/null; then
  log "  백엔드(8000) 무응답 → compose up -d"
  $COMPOSE -f "$COMPOSE_FILE" up -d >> "$LOG" 2>&1
  for i in $(seq 1 30); do curl -s -m 4 -o /dev/null "$LOCAL_URL" 2>/dev/null && { log "  백엔드 복구됨(${i}x3s)"; break; }; sleep 3; done
fi

# 3) 로컬은 정상인데 공개 URL 실패 → Tailscale Funnel 재활성화
if curl -s -m 5 -o /dev/null "$LOCAL_URL" 2>/dev/null; then
  pcode=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$PUBLIC_URL" 2>/dev/null)
  if [ "$pcode" != "200" ]; then
    log "  백엔드 정상이나 공개 URL 실패(code=$pcode) → Tailscale Funnel 재활성화"
    "$TS" funnel --bg 8000 >> "$LOG" 2>&1
  fi
fi

# 4) 사이드카(STT/TTS) 점검 → launchd kickstart
if ! curl -s -m 3 -o /dev/null http://localhost:8181/health 2>/dev/null; then
  log "  TTS(8181) 무응답 → melo-sidecar 재시작"
  launchctl kickstart -k "gui/${UID_NUM}/com.healthcare-rag.melo-sidecar" 2>> "$LOG"
fi
if ! curl -s -m 3 -o /dev/null http://localhost:8182/health 2>/dev/null; then
  log "  STT(8182) 무응답 → stt-sidecar 재시작"
  launchctl kickstart -k "gui/${UID_NUM}/com.healthcare-rag.stt-sidecar" 2>> "$LOG"
fi

# 최종 상태 한 줄
final=$(curl -s -m 8 -H "ngrok-skip-browser-warning: true" -o /dev/null -w '%{http_code}' "$PUBLIC_URL" 2>/dev/null)
log "복구 점검 완료 — 공개 URL code=$final"
