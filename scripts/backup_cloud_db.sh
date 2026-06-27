#!/usr/bin/env bash
# Cloud SQL(pgvector) DB를 로컬에 백업한다 (pg_dump, custom format).
# 비밀번호가 들어있는 DATABASE_URL을 .env.local에서 읽어 그대로 pg_dump에 넘기므로,
# 반드시 본인 터미널에서 직접 실행할 것 (에이전트가 대신 실행하지 않음).
#
# 사용법: bash scripts/backup_cloud_db.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BACKUP_DIR="$HOME/Backups/rag-healthcare-db"
mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"

DATABASE_URL=$(grep '^DATABASE_URL=' .env.local | cut -d= -f2-)
if [ -z "$DATABASE_URL" ]; then
  echo "❌ .env.local에서 DATABASE_URL을 찾지 못했습니다." >&2
  exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="$BACKUP_DIR/healthcare_${STAMP}.dump"

echo "📦 백업 중 → $OUT"
docker run --rm postgres:15 pg_dump "$DATABASE_URL" \
  --no-owner --no-privileges --format=custom \
  > "$OUT"

SIZE=$(du -h "$OUT" | cut -f1)
echo "✅ 백업 완료: $OUT ($SIZE)"
echo "   복원 확인: docker run --rm -v \"$OUT:/b.dump:ro\" postgres:15 pg_restore --list /b.dump | head -20"
