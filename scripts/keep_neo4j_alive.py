#!/usr/bin/env python3
"""
Neo4j AuraDB Free 인스턴스 활성 유지 스크립트
72시간 미사용 시 자동 pause되는 것을 방지하기 위해 주기적으로 실행한다.
크론 등록: crontab -e → 0 */12 * * * /path/to/python /path/to/scripts/keep_neo4j_alive.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# .env.local 로드
from dotenv import load_dotenv
load_dotenv(project_root / ".env.local")
load_dotenv(project_root / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_PASSWORD]):
    print("❌ NEO4J_URI 또는 NEO4J_PASSWORD 환경변수가 없습니다.")
    sys.exit(1)

try:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run("RETURN 1 AS alive")
        record = result.single()
    driver.close()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] ✅ Neo4j AuraDB 활성 확인 완료")
    sys.exit(0)

except Exception as e:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] ❌ Neo4j 연결 실패: {e}")
    sys.exit(1)
