"""
Neo4j 기반 건강 도메인 지식그래프 (Health Knowledge Graph)

Neo4j 그래프DB를 사용하여
증상 → 질환 → 치료/관리 → 주의사항 관계를 모델링한다.

노드 라벨: Symptom, Condition, Treatment, BodyPart, RiskFactor, Medication, Lifestyle
관계 타입: SYMPTOM_OF, TREATS, AFFECTS, CAUSES, PREVENTS, MANAGED_BY, RELATED_TO
"""

from typing import Optional

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


# ── Neo4j 노드 라벨 (Cypher 쿼리에서 사용) ──
class NodeLabel:
    SYMPTOM = "Symptom"
    CONDITION = "Condition"
    TREATMENT = "Treatment"
    BODY_PART = "BodyPart"
    RISK_FACTOR = "RiskFactor"
    MEDICATION = "Medication"
    LIFESTYLE = "Lifestyle"


# ── Neo4j 관계 타입 ──
class RelType:
    INDICATES = "INDICATES"
    TREATS = "TREATS"
    AFFECTS = "AFFECTS"
    CAUSES = "CAUSES"
    PREVENTS = "PREVENTS"
    SYMPTOM_OF = "SYMPTOM_OF"
    MANAGED_BY = "MANAGED_BY"
    RELATED_TO = "RELATED_TO"


class Neo4jHealthKG:
    """Neo4j 기반 건강 도메인 지식그래프"""

    def __init__(self):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        )
        self._database = settings.NEO4J_DATABASE
        self._verify_connection()
        self._ensure_indexes()

    def _verify_connection(self):
        """Neo4j 연결 확인"""
        try:
            self.driver.verify_connectivity()
            stats = self.get_stats()
            logger.info(
                f"🧠 Neo4j 지식그래프 연결 성공 | "
                f"uri={settings.NEO4J_URI} | "
                f"노드={stats['node_count']}, 엣지={stats['edge_count']}"
            )
        except Exception as e:
            logger.error(f"Neo4j 연결 실패: {e}")
            raise

    def _ensure_indexes(self):
        """필수 인덱스 생성 (멱등)"""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Condition) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Symptom) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Treatment) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:BodyPart) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:RiskFactor) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Lifestyle) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Medication) ON (n.name)",
        ]
        with self.driver.session(database=self._database) as session:
            for q in index_queries:
                session.run(q)

    def close(self):
        """드라이버 종료"""
        self.driver.close()

    # ═══════════════════════════════════════════
    # 통계/인트로스펙션
    # ═══════════════════════════════════════════

    def get_stats(self) -> dict:
        """노드/엣지 수 반환"""
        with self.driver.session(database=self._database) as session:
            node_count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            edge_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        return {"node_count": node_count, "edge_count": edge_count}

    def get_all_node_names(self) -> list[str]:
        """모든 노드의 name 속성 목록 반환 (키워드 매칭용)"""
        with self.driver.session(database=self._database) as session:
            result = session.run("MATCH (n) WHERE n.name IS NOT NULL RETURN n.name AS name")
            return [record["name"] for record in result]

    def get_node_names_by_label(self, label: str) -> list[str]:
        """특정 라벨의 노드명 반환"""
        with self.driver.session(database=self._database) as session:
            result = session.run(
                f"MATCH (n:{label}) WHERE n.name IS NOT NULL RETURN n.name AS name"
            )
            return [record["name"] for record in result]

    # ═══════════════════════════════════════════
    # 검색 API
    # ═══════════════════════════════════════════

    def get_condition_info(self, condition_name: str) -> dict:
        """
        질환명으로 증상, 치료법, 주의사항 등 종합 정보를 가져온다.
        """
        query = """
        MATCH (c:Condition {name: $name})
        OPTIONAL MATCH (s)-[:SYMPTOM_OF]->(c)
        OPTIONAL MATCH (c)-[:MANAGED_BY]->(t)
        OPTIONAL MATCH (c)-[:AFFECTS]->(bp:BodyPart)
        OPTIONAL MATCH (rf)-[:CAUSES]->(c)
        OPTIONAL MATCH (prev)-[:PREVENTS]->(c)
        RETURN c.name AS condition,
               c.description AS description,
               collect(DISTINCT {name: s.name, desc: s.description}) AS symptoms,
               collect(DISTINCT {name: t.name, desc: t.description}) AS treatments,
               collect(DISTINCT bp.name) AS body_parts,
               collect(DISTINCT {name: rf.name, desc: rf.description}) AS risk_factors,
               collect(DISTINCT {name: prev.name, desc: prev.description}) AS prevention
        """
        with self.driver.session(database=self._database) as session:
            result = session.run(query, name=condition_name).single()

        if not result or not result["condition"]:
            return {}

        return {
            "condition": result["condition"],
            "description": result["description"] or "",
            "symptoms": [s for s in result["symptoms"] if s["name"]],
            "treatments": [t for t in result["treatments"] if t["name"]],
            "body_parts": [bp for bp in result["body_parts"] if bp],
            "risk_factors": [r for r in result["risk_factors"] if r["name"]],
            "prevention": [p for p in result["prevention"] if p["name"]],
        }

    def get_symptom_conditions(self, symptom: str) -> list[str]:
        """증상으로부터 가능한 질환 목록을 반환한다."""
        query = """
        MATCH (s:Symptom)-[:SYMPTOM_OF]->(c:Condition)
        WHERE s.name = $symptom OR s.name CONTAINS $symptom
        RETURN DISTINCT c.name AS condition
        """
        with self.driver.session(database=self._database) as session:
            result = session.run(query, symptom=symptom)
            return [record["condition"] for record in result]

    def find_related_nodes(self, keyword: str, depth: int = 2) -> list[dict]:
        """
        키워드와 관련된 노드들을 가변 길이 경로로 탐색한다 (depth홉까지).
        """
        query = """
        MATCH (start)
        WHERE start.name CONTAINS $keyword
           OR (start.description IS NOT NULL AND start.description CONTAINS $keyword)
        WITH start LIMIT 5
        MATCH path = (start)-[*1..%d]-(related)
        WHERE related <> start
        RETURN DISTINCT related.name AS name,
               labels(related)[0] AS type,
               related.description AS desc,
               length(path) AS distance
        ORDER BY distance
        LIMIT 20
        """ % min(depth, 3)  # 최대 3홉으로 제한

        with self.driver.session(database=self._database) as session:
            result = session.run(query, keyword=keyword)
            nodes = [dict(record) for record in result]

        return [n for n in nodes if n["name"]]

    # ═══════════════════════════════════════════
    # 쓰기 API (파이프라인용)
    # ═══════════════════════════════════════════

    def clear_all(self):
        """모든 노드/관계 삭제 (주의: 전체 초기화)"""
        with self.driver.session(database=self._database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.warning("⚠️ Neo4j 그래프 전체 초기화됨")

    def merge_node(self, name: str, label: str, description: str = ""):
        """노드를 MERGE (없으면 생성, 있으면 업데이트)"""
        query = f"""
        MERGE (n:{label} {{name: $name}})
        SET n.description = $desc
        """
        with self.driver.session(database=self._database) as session:
            session.run(query, name=name, desc=description)

    def merge_relationship(self, src: str, dst: str, rel_type: str):
        """관계를 MERGE (중복 방지)"""
        query = f"""
        MATCH (a {{name: $src}})
        MATCH (b {{name: $dst}})
        MERGE (a)-[:{rel_type}]->(b)
        """
        with self.driver.session(database=self._database) as session:
            session.run(query, src=src, dst=dst)

    def bulk_import_graph_documents(self, graph_documents: list):
        """
        LangChain GraphDocument 리스트를 Neo4j에 벌크 적재.
        langchain_neo4j.Neo4jGraph.add_graph_documents() 위임.
        """
        from langchain_neo4j import Neo4jGraph

        neo4j_graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            database=self._database,
        )
        neo4j_graph.add_graph_documents(graph_documents, baseEntityLabel=True)
        neo4j_graph._driver.close()
        logger.info(f"📥 GraphDocument {len(graph_documents)}개 벌크 적재 완료")


# ── 싱글톤 ──
_neo4j_kg: Optional[Neo4jHealthKG] = None


def get_neo4j_kg() -> Neo4jHealthKG:
    """Neo4j 지식그래프 싱글톤 반환"""
    global _neo4j_kg
    if _neo4j_kg is None:
        _neo4j_kg = Neo4jHealthKG()
    return _neo4j_kg
