"""
GraphRAG 검색기
지식그래프 + 벡터 검색을 결합하여 구조화된 컨텍스트를 생성한다.

1. 쿼리에서 건강 관련 키워드 추출
2. 지식그래프에서 관련 노드 탐색 (증상→질환→치료)
3. 구조화된 텍스트로 변환하여 LLM 프롬프트에 삽입
"""

import re
from typing import Optional

from app.knowledge_graph.health_kg import (
    HealthKnowledgeGraph,
    get_health_kg,
    NodeType,
)
from app.logger import get_logger

logger = get_logger(__name__)


class GraphRAGRetriever:
    """지식그래프 기반 RAG 검색기"""

    def __init__(self, kg: Optional[HealthKnowledgeGraph] = None):
        self.kg = kg or get_health_kg()

    def search(self, query: str) -> str:
        """
        쿼리에서 키워드를 추출하고 지식그래프를 탐색하여
        구조화된 컨텍스트 문자열을 반환한다.

        Args:
            query: 사용자 질문 또는 재작성된 쿼리

        Returns:
            지식그래프 기반 컨텍스트 문자열 (없으면 빈 문자열)
        """
        keywords = self._extract_health_keywords(query)

        if not keywords:
            return ""

        context_parts = []

        for keyword in keywords:
            # 1. 증상→질환 경로 탐색
            conditions = self.kg.get_symptom_conditions(keyword)
            for cond in conditions:
                info = self.kg.get_condition_info(cond)
                if info:
                    context_parts.append(self._format_condition_info(info))

            # 2. 질환 직접 매칭
            info = self.kg.get_condition_info(keyword)
            if info and info.get("symptoms"):
                context_parts.append(self._format_condition_info(info))

            # 3. 관련 노드 탐색 (깊이 2)
            if not conditions and not info.get("symptoms", []):
                related = self.kg.find_related_nodes(keyword, depth=2)
                if related:
                    summary = self._format_related_nodes(keyword, related)
                    if summary:
                        context_parts.append(summary)

        # 중복 제거
        unique_parts = list(dict.fromkeys(context_parts))
        result = "\n".join(unique_parts[:3])  # 최대 3개 질환 정보

        if result:
            logger.info(f"🧠 GraphRAG | keywords={keywords} | context_len={len(result)}")

        return result

    def _extract_health_keywords(self, query: str) -> list[str]:
        """쿼리에서 건강 관련 키워드를 추출한다."""
        keywords = []

        # 지식그래프의 모든 노드명과 매칭
        for node in self.kg.graph.nodes:
            if node in query:
                keywords.append(node)

        # 추가 패턴 매칭 (일상 표현 → 의학 용어)
        colloquial_map = {
            r"잠을?\s*못": "불면증",
            r"잠이?\s*안": "불면증",
            r"밤에?\s*깨": "수면 패턴 변화",
            r"발톱.*(안|안쪽|파고|들어)": "내향성 발톱",
            r"발톱.*(두꺼|변형|휘)": "발톱 변형",
            r"소변.*(못 참|실수|자주)": "요실금",
            r"머리.*(빠지|빠져|탈모)": "머리카락 빠짐",
            r"피부.*(가려|건조|트러블)": "피부 가려움",
            r"숨.*(차|안 쉬|못 쉬)": "호흡곤란",
            r"기침.*(오래|안 멈|계속)": "만성기침",
            r"손.*(저리|감각)": "손발 저림",
            r"발.*(저리|감각)": "손발 저림",
            r"귀.*(안 들|잘 안)": "소리 안 들림",
            r"눈.*(침침|흐릿|안 보)": "시력 저하",
            r"잇몸.*(피|출혈|붓)": "잇몸 출혈",
            r"변.*(안 나|못 보|힘들)": "배변 곤란",
            r"배.*(더부룩|팽만|부른)": "복부 팽만",
            r"기운.*(없|빠지|저하)": "기력 저하",
            r"뼈.*(아프|통증|쑤시)": "뼈 통증",
            r"혈당.*(높|올라)": "고혈당",
            r"혈당.*(낮|떨어)": "저혈당",
        }

        for pattern, medical_term in colloquial_map.items():
            if re.search(pattern, query):
                if medical_term not in keywords:
                    keywords.append(medical_term)

        return keywords[:5]  # 최대 5개

    def _format_condition_info(self, info: dict) -> str:
        """질환 정보를 읽기 쉬운 텍스트로 변환한다."""
        parts = [f"▶ {info['condition']}: {info.get('description', '')}"]

        symptoms = info.get("symptoms", [])
        if symptoms:
            symptom_names = [s["name"] for s in symptoms]
            parts.append(f"  증상: {', '.join(symptom_names)}")

        treatments = info.get("treatments", [])
        if treatments:
            for t in treatments:
                parts.append(f"  관리: {t['name']} — {t.get('desc', '')}")

        prevention = info.get("prevention", [])
        if prevention:
            prev_names = [p["name"] for p in prevention]
            parts.append(f"  예방: {', '.join(prev_names)}")

        risk_factors = info.get("risk_factors", [])
        if risk_factors:
            risk_names = [r["name"] for r in risk_factors]
            parts.append(f"  위험요인: {', '.join(risk_names)}")

        return "\n".join(parts)

    def _format_related_nodes(self, keyword: str, nodes: list[dict]) -> str:
        """관련 노드 목록을 텍스트로 변환한다."""
        if not nodes:
            return ""

        # 타입별 그룹핑
        by_type = {}
        for node in nodes:
            ntype = node.get("type", "unknown")
            if ntype not in by_type:
                by_type[ntype] = []
            by_type[ntype].append(node)

        parts = [f"▶ '{keyword}' 관련 정보:"]

        type_labels = {
            NodeType.CONDITION: "관련 질환",
            NodeType.SYMPTOM: "관련 증상",
            NodeType.TREATMENT: "관리 방법",
            NodeType.LIFESTYLE: "생활 습관",
            NodeType.RISK_FACTOR: "주의 사항",
        }

        for ntype, label in type_labels.items():
            group = by_type.get(ntype, [])
            if group:
                names = [f"{n['name']}({n.get('desc', '')})" for n in group[:3]]
                parts.append(f"  {label}: {', '.join(names)}")

        return "\n".join(parts) if len(parts) > 1 else ""


# 싱글톤
_graph_rag: Optional[GraphRAGRetriever] = None


def get_graph_rag() -> GraphRAGRetriever:
    global _graph_rag
    if _graph_rag is None:
        _graph_rag = GraphRAGRetriever()
    return _graph_rag
