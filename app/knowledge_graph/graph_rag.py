"""
GraphRAG 검색기 (Neo4j 기반)
Neo4j 지식그래프 + 벡터 검색을 결합하여 구조화된 컨텍스트를 생성한다.

키워드 추출 3단계 하이브리드:
  1단계: 규칙 기반 — 그래프 노드명 직접매칭 + 구어→의학용어 regex (즉시, 0ms)
  2단계: 임베딩 유사도 — sentence-transformer로 노드 의미 매칭 (+30ms)
  3단계: LLM Fallback — 1~2단계 결과 부족 시 Ollama에 의학 키워드 추출 요청 (+500ms~2s)
"""

import re
from typing import Optional

import numpy as np

from app.knowledge_graph.health_kg import Neo4jHealthKG, get_neo4j_kg, NodeLabel
from app.config import settings
from app.logger import get_logger
from app.model.local_model import get_embedding_model

logger = get_logger(__name__)

# ── 임베딩 유사도 매칭 설정 ──
EMBEDDING_SIMILARITY_THRESHOLD = 0.55
EMBEDDING_TOP_K = 2
EMBEDDING_GAP_THRESHOLD = 0.04


class GraphRAGRetriever:
    """Neo4j 지식그래프 기반 RAG 검색기 (하이브리드 키워드 추출)"""

    def __init__(self, kg: Optional[Neo4jHealthKG] = None):
        self.kg = kg or get_neo4j_kg()
        # ── 임베딩 인덱스 (앱 시작 시 1회 구축) ──
        self._node_names: list[str] = []
        self._node_embeddings: Optional[np.ndarray] = None
        self._embedding = get_embedding_model()
        self._build_node_index()

    def _build_node_index(self):
        """Neo4j의 모든 노드명을 임베딩하여 유사도 검색 인덱스를 구축한다."""
        try:
            node_names = self.kg.get_all_node_names()
            if not node_names:
                logger.warning("Neo4j에 노드가 없습니다. build_neo4j_kg.py를 먼저 실행하세요.")
                return

            self._node_names = node_names
            self._node_embeddings = self._embedding.embed(node_names)
            logger.info(
                f"🔗 GraphRAG 노드 임베딩 인덱스 구축 완료: "
                f"{len(self._node_names)}개 노드, "
                f"model={settings.EMBEDDING_MODEL}, "
                f"dim={self._node_embeddings.shape[1]}"
            )
        except Exception as e:
            logger.warning(f"노드 임베딩 인덱스 구축 실패 (규칙 기반+LLM만 사용): {e}")
            self._node_embeddings = None

    def rebuild_index(self):
        """노드 임베딩 인덱스를 재구축한다 (그래프 변경 후 호출)."""
        self._node_names = []
        self._node_embeddings = None
        self._build_node_index()

    # ═══════════════════════════════════════════
    # 검색 메인 엔트리
    # ═══════════════════════════════════════════

    def search(self, query: str) -> str:
        """
        쿼리에서 키워드를 추출하고 Neo4j 지식그래프를 탐색하여
        구조화된 컨텍스트 문자열을 반환한다.
        """
        keywords = self._extract_keywords_hybrid(query)

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
            if not conditions and not (info and info.get("symptoms")):
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

    # ═══════════════════════════════════════════
    # 하이브리드 키워드 추출 (3단계)
    # ═══════════════════════════════════════════

    def _extract_keywords_hybrid(self, query: str) -> list[str]:
        """
        3단계 하이브리드 키워드 추출:
        1) 규칙 기반 (직접매칭 + regex)  — ~0ms
        2) 임베딩 유사도 매칭            — ~30ms
        3) LLM Fallback                 — ~500ms~2s (필요시만)
        """
        keywords = []

        # ── Stage 1: 규칙 기반 ──
        rule_keywords = self._extract_by_rules(query)
        keywords.extend(rule_keywords)

        # ── Stage 2: 임베딩 유사도 매칭 ──
        embed_keywords = self._extract_by_embedding(query)
        for kw in embed_keywords:
            if kw not in keywords:
                keywords.append(kw)

        # ── Stage 3: LLM Fallback (1~2단계 결과가 부족할 때만) ──
        if len(keywords) == 0:
            llm_keywords = self._extract_by_llm(query)
            for kw in llm_keywords:
                if kw not in keywords:
                    keywords.append(kw)

        if keywords:
            logger.info(
                f"🔑 키워드 추출 | "
                f"rule={rule_keywords} | "
                f"embed={embed_keywords} | "
                f"final={keywords[:5]}"
            )

        return keywords[:5]

    # ── Stage 1: 규칙 기반 ──

    def _extract_by_rules(self, query: str) -> list[str]:
        """그래프 노드명 직접매칭 + 구어→의학용어 regex 패턴."""
        keywords = []

        # Neo4j에서 노드명 가져와서 직접 매칭
        for name in self._node_names:
            if name in query:
                keywords.append(name)

        # 구어 → 의학용어 매핑
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

        return keywords

    # ── Stage 2: 임베딩 유사도 매칭 ──

    def _extract_by_embedding(self, query: str) -> list[str]:
        """
        쿼리 임베딩과 그래프 노드 임베딩 간 코사인 유사도로 관련 노드를 찾는다.
        """
        if self._node_embeddings is None or len(self._node_names) == 0:
            return []

        try:
            query_emb = self._embedding.embed([query])  # shape: (1, dim)

            # 코사인 유사도 계산
            query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
            node_norms = self._node_embeddings / (
                np.linalg.norm(self._node_embeddings, axis=1, keepdims=True) + 1e-10
            )
            similarities = (query_norm @ node_norms.T).flatten()

            sorted_indices = np.argsort(similarities)[::-1]
            top_indices = sorted_indices[:EMBEDDING_TOP_K]

            # 상대 갭 체크
            if len(sorted_indices) > 3:
                gap = float(similarities[sorted_indices[0]] - similarities[sorted_indices[3]])
                if gap < EMBEDDING_GAP_THRESHOLD:
                    logger.debug(
                        f"임베딩 갭 부족: top1={similarities[sorted_indices[0]]:.3f}, "
                        f"top4={similarities[sorted_indices[3]]:.3f}, gap={gap:.3f}"
                    )
                    return []

            matched = []
            for idx in top_indices:
                score = similarities[idx]
                if score >= EMBEDDING_SIMILARITY_THRESHOLD:
                    node_name = self._node_names[idx]
                    matched.append((node_name, float(score)))

            if matched:
                match_str = ", ".join(
                    f"{name}({score:.2f})" for name, score in matched
                )
                logger.debug(f"🔍 임베딩 매칭: query='{query}' → {match_str}")

            return [name for name, _ in matched]

        except Exception as e:
            logger.debug(f"임베딩 매칭 실패: {e}")
            return []

    # ── Stage 3: LLM Fallback ──

    def _extract_by_llm(self, query: str) -> list[str]:
        """
        규칙+임베딩으로 키워드를 못 찾았을 때 LLM에게 의학 키워드 추출을 요청한다.
        """
        try:
            import httpx

            node_list = ", ".join(self._node_names[:50])

            prompt = (
                "당신은 의료 키워드 추출기입니다.\n"
                "아래 사용자 메시지에서 건강/의료 관련 키워드를 추출하세요.\n"
                "반드시 아래 후보 목록에 있는 단어만 선택하세요.\n"
                "해당 없으면 '없음'이라고만 답하세요.\n\n"
                f"[후보 목록]\n{node_list}\n\n"
                f"[사용자 메시지]\n{query}\n\n"
                "[추출된 키워드 (쉼표 구분)]"
            )

            response = httpx.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 50,
                    },
                },
                timeout=5.0,
            )
            response.raise_for_status()

            result = response.json().get("response", "").strip()

            if "없음" in result or not result:
                return []

            # <think>...</think> 블록 제거
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL)
            result = re.sub(r"</?think>", "", result).strip()

            # 쉼표로 분리 후 실제 그래프 노드인지 검증
            candidates = [kw.strip() for kw in result.split(",")]
            node_name_set = set(self._node_names)
            valid_keywords = [kw for kw in candidates if kw in node_name_set]

            if valid_keywords:
                logger.info(f"🤖 LLM 키워드 추출: '{query}' → {valid_keywords}")

            return valid_keywords[:3]

        except Exception as e:
            logger.debug(f"LLM 키워드 추출 스킵: {e}")
            return []

    # ═══════════════════════════════════════════
    # 포매팅
    # ═══════════════════════════════════════════

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
            NodeLabel.CONDITION: "관련 질환",
            NodeLabel.SYMPTOM: "관련 증상",
            NodeLabel.TREATMENT: "관리 방법",
            NodeLabel.LIFESTYLE: "생활 습관",
            NodeLabel.RISK_FACTOR: "주의 사항",
        }

        for ntype, label in type_labels.items():
            group = by_type.get(ntype, [])
            if group:
                names = [f"{n['name']}({n.get('desc', '')})" for n in group[:3]]
                parts.append(f"  {label}: {', '.join(names)}")

        return "\n".join(parts) if len(parts) > 1 else ""


# ── 싱글톤 ──
_graph_rag: Optional[GraphRAGRetriever] = None


def get_graph_rag() -> GraphRAGRetriever:
    global _graph_rag
    if _graph_rag is None:
        _graph_rag = GraphRAGRetriever()
    return _graph_rag
