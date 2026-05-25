"""
대화 기록 추출·분석 스크립트 (연구/이용자 그룹 분석용)

Cloud SQL(pgvector와 동일 DB)에 저장된 대화를 두 소스에서 뽑아 CSV로 내보내고,
이용자/인텐트/위험도/시간대별 요약 통계를 출력한다.

소스:
- conversation_logs : 본 프로젝트가 턴마다 남기는 분석용 로그
                      (KST 타임스탬프 + intent/risk_level/detected_symptoms 등 메타데이터)
- chat_history      : LangChain이 자동 생성하는 런타임 대화 테이블
                      (session_id=nickname, message=JSONB, 타임스탬프 없음 → 순서만 보존)

사용 예:
    python scripts/export_conversations.py                       # 전체 추출 + 요약
    python scripts/export_conversations.py --out-dir ./export    # 출력 폴더 지정
    python scripts/export_conversations.py --nickname 김할머니    # 특정 이용자만
    python scripts/export_conversations.py --summary-only        # CSV 없이 통계만
"""

import argparse
import asyncio
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 백엔드와 동일한 환경변수 로드 (.env.local 우선, .env 폴백 — deploy_local.sh 규칙).
# app.config import 전에 os.environ에 주입해야 pydantic Settings가 DATABASE_URL을 인식한다.
try:
    from dotenv import load_dotenv

    load_dotenv(project_root / ".env", override=False)
    load_dotenv(project_root / ".env.local", override=True)  # .env.local이 우선
except ImportError:
    pass

import asyncpg

from app.config import settings
from app.logger import get_logger
from app.utils.timezone import KST, get_kst_now

logger = get_logger(__name__)


def _connect_params(connection_string: str) -> dict:
    """asyncpg 연결 파라미터 (Cloud SQL Unix 소켓 / TCP 모두 지원).

    app/langchain_store.py 의 _parse_connection_for_asyncpg 와 동일 규칙."""
    if "?host=/cloudsql/" in connection_string:
        match = re.match(
            r"postgresql://([^:]+):([^@]+)@/([^?]+)\?host=(.+)",
            connection_string,
        )
        if match:
            return {
                "user": match.group(1),
                "password": match.group(2),
                "database": match.group(3),
                "host": match.group(4),
            }
    return {"dsn": connection_string}


async def _table_exists(conn: asyncpg.Connection, table: str) -> bool:
    return bool(await conn.fetchval("SELECT to_regclass($1)", table))


# ==========================================
# conversation_logs (분석용 로그)
# ==========================================

async def fetch_conversation_logs(conn, nickname: str | None) -> list[dict]:
    if not await _table_exists(conn, "conversation_logs"):
        logger.warning(
            "conversation_logs 테이블이 없습니다. "
            "앱을 한 번 띄워 스키마를 생성하거나 새 대화를 쌓은 뒤 다시 실행하세요."
        )
        return []

    where = "WHERE nickname = $1" if nickname else ""
    args = [nickname] if nickname else []
    rows = await conn.fetch(
        f"""
        SELECT id, nickname, user_message, ai_message,
               intent, intent_confidence, risk_level,
               detected_symptoms, risk_categories,
               repeated_question, topic_drifted, created_at
        FROM conversation_logs
        {where}
        ORDER BY created_at ASC, id ASC
        """,
        *args,
    )

    out = []
    for r in rows:
        created = r["created_at"]
        created_kst = created.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S") if created else ""
        out.append(
            {
                "id": r["id"],
                "nickname": r["nickname"],
                "created_at_kst": created_kst,
                "intent": r["intent"] or "",
                "intent_confidence": r["intent_confidence"] if r["intent_confidence"] is not None else "",
                "risk_level": r["risk_level"] or "",
                "detected_symptoms": r["detected_symptoms"] or "[]",
                "risk_categories": r["risk_categories"] or "[]",
                "repeated_question": r["repeated_question"],
                "topic_drifted": r["topic_drifted"],
                "user_message": (r["user_message"] or "").replace("\n", " ").strip(),
                "ai_message": (r["ai_message"] or "").replace("\n", " ").strip(),
            }
        )
    return out


# ==========================================
# chat_history (LangChain 레거시 JSONB)
# ==========================================

async def fetch_chat_history(conn, nickname: str | None) -> list[dict]:
    if not await _table_exists(conn, "chat_history"):
        logger.warning("chat_history 테이블이 없습니다. 건너뜁니다.")
        return []

    where = "WHERE session_id = $1" if nickname else ""
    args = [nickname] if nickname else []
    rows = await conn.fetch(
        f"SELECT id, session_id, message FROM chat_history {where} ORDER BY id ASC",
        *args,
    )

    out = []
    for r in rows:
        msg = r["message"]
        if isinstance(msg, str):
            try:
                msg = json.loads(msg)
            except json.JSONDecodeError:
                msg = {}
        mtype = msg.get("type", "")
        content = (msg.get("data") or {}).get("content", "")
        out.append(
            {
                "id": r["id"],
                "nickname": r["session_id"],
                "role": "user" if mtype == "human" else ("assistant" if mtype == "ai" else mtype),
                "content": (content or "").replace("\n", " ").strip(),
            }
        )
    return out


# ==========================================
# CSV 쓰기
# ==========================================

def write_csv(rows: list[dict], path: Path):
    if not rows:
        logger.info(f"  (빈 데이터 — {path.name} 생략)")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # 엑셀 한글 호환 BOM
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"  ✅ {path}  ({len(rows)} rows)")


# ==========================================
# 요약 통계
# ==========================================

def print_summary(logs: list[dict], history: list[dict]):
    print("\n" + "=" * 60)
    print("📊 대화 기록 요약")
    print("=" * 60)

    # chat_history 기준 (가장 포괄적 — 모든 런타임 대화)
    hist_users = Counter(h["nickname"] for h in history)
    hist_turns = sum(1 for h in history if h["role"] == "user")
    print(f"\n[chat_history] 런타임 대화")
    print(f"  - 이용자 수(고유 nickname): {len(hist_users)}")
    print(f"  - 사용자 발화 턴 수       : {hist_turns}")
    print(f"  - 총 메시지 수            : {len(history)}")

    print(f"\n[conversation_logs] 분석용 로그 (타임스탬프·메타데이터 포함)")
    if not logs:
        print("  - 아직 로그 없음 (이번 변경 배포 이후 쌓이는 대화부터 기록됩니다)")
        print("=" * 60 + "\n")
        return

    print(f"  - 기록된 턴 수: {len(logs)}")
    dates = [l["created_at_kst"][:10] for l in logs if l["created_at_kst"]]
    if dates:
        print(f"  - 기간       : {min(dates)} ~ {max(dates)} (KST)")

    # 이용자별
    by_user = Counter(l["nickname"] for l in logs)
    print(f"\n  ▸ 이용자별 턴 수 (상위 10):")
    for name, cnt in by_user.most_common(10):
        print(f"      {name}: {cnt}")

    # 인텐트 분포
    by_intent = Counter(l["intent"] for l in logs if l["intent"])
    print(f"\n  ▸ 인텐트 분포:")
    total = sum(by_intent.values()) or 1
    for intent, cnt in by_intent.most_common():
        print(f"      {intent:16s} {cnt:5d}  ({cnt/total*100:4.1f}%)")

    # 위험도 분포
    by_risk = Counter(l["risk_level"] for l in logs if l["risk_level"])
    print(f"\n  ▸ 위험도(risk_level) 분포:")
    for risk, cnt in by_risk.most_common():
        print(f"      {risk:10s} {cnt}")

    # 흐름 플래그
    repeated = sum(1 for l in logs if l["repeated_question"])
    drifted = sum(1 for l in logs if l["topic_drifted"])
    print(f"\n  ▸ 반복 질문 감지: {repeated} | 주제 이탈 감지: {drifted}")
    print("=" * 60 + "\n")


# ==========================================
# 메인
# ==========================================

async def run(out_dir: Path, nickname: str | None, summary_only: bool):
    db_url = settings.database_url
    if not db_url:
        logger.error("DATABASE_URL이 설정되지 않았습니다. .env / .env.local을 확인하세요.")
        return

    logger.info(f"DB 연결 중... (nickname 필터: {nickname or '없음 — 전체'})")
    conn = await asyncpg.connect(**_connect_params(db_url), timeout=15)
    try:
        logs = await fetch_conversation_logs(conn, nickname)
        history = await fetch_chat_history(conn, nickname)
    finally:
        await conn.close()

    if not summary_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = get_kst_now().strftime("%Y%m%d_%H%M")
        logger.info(f"CSV 내보내기 → {out_dir}/")
        write_csv(logs, out_dir / f"conversation_logs_{stamp}.csv")
        write_csv(history, out_dir / f"chat_history_{stamp}.csv")

    print_summary(logs, history)


def main():
    parser = argparse.ArgumentParser(description="대화 기록 추출·분석")
    parser.add_argument("--out-dir", default="export", help="CSV 출력 폴더 (기본: ./export)")
    parser.add_argument("--nickname", default=None, help="특정 이용자만 추출")
    parser.add_argument("--summary-only", action="store_true", help="CSV 없이 요약 통계만 출력")
    args = parser.parse_args()

    asyncio.run(run(Path(args.out_dir), args.nickname, args.summary_only))


if __name__ == "__main__":
    main()
