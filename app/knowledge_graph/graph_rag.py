"""
GraphRAG 검색기 (Hybrid Keyword Extraction)
지식그래프 + 벡터 검색을 결합하여 구조화된 컨텍스트를 생성한다.

키워드 추출 3단계 하이브리드:
  1단계: 규칙 기반 — 그래프 노드명 직접매칭 + 구어→의학용어 regex (즉시, 0ms)
  2단계: 임베딩 유사도 — sentence-transformer로 노드 의미 매칭 (+30ms)
  3단계: LLM Fallback — 1~2단계 결과 부족 시 Ollama에 의학 키워드 추출 요청 (+500ms~2s)
"""

import asyncio
import re
from typing import Optional

import numpy as np

from app.knowledge_graph.health_kg import (
    HealthKnowledgeGraph,
    get_health_kg,
    NodeType,
)
from app.config import settings
from app.logger import get_logger
from app.model.local_model import get_embedding_model

logger = get_logger(__name__)

# ── 임베딩 유사도 매칭 설정 ──
# 다국어 모델용 임계값
EMBEDDING_SIMILARITY_THRESHOLD = 0.55   # 절대 임계값
EMBEDDING_TOP_K = 2                     # 상위 K개까지 후보 선정
EMBEDDING_GAP_THRESHOLD = 0.04          # 1위와 4위 간 최소 점수 갭 (작으면 노이즈)


class GraphRAGRetriever:
    """지식그래프 기반 RAG 검색기 (하이브리드 키워드 추출)"""

    def __init__(self, kg: Optional[HealthKnowledgeGraph] = None):
        self.kg = kg or get_health_kg()
        # ── 임베딩 인덱스 (앱 시작 시 1회 구축) ──
        self._node_names: list[str] = []
        self._node_embeddings: Optional[np.ndarray] = None
        self._embedding = get_embedding_model()  # 기존 싱글톤 재사용
        self._build_node_index()

    def _build_node_index(self):
        """지식그래프의 모든 노드를 임베딩하여 유사도 검색 인덱스를 구축한다."""
        try:
            # 노드명만 임베딩 (설명 포함 시 일반 단어가 노이즈 유발)
            node_texts = []
            for node, data in self.kg.graph.nodes(data=True):
                node_texts.append(node)
                self._node_names.append(node)

            self._node_embeddings = self._embedding.embed(node_texts)
            logger.info(
                f"🔗 GraphRAG 노드 임베딩 인덱스 구축 완료: "
                f"{len(self._node_names)}개 노드, "
                f"model={settings.EMBEDDING_MODEL}, "
                f"dim={self._node_embeddings.shape[1]}"
            )
        except Exception as e:
            logger.warning(f"노드 임베딩 인덱스 구축 실패 (규칙 기반+LLM만 사용): {e}")
            self._node_embeddings = None

    # ═══════════════════════════════════════════
    # 검색 메인 엔트리
    # ═══════════════════════════════════════════

    def search(self, query: str) -> str:
        """
        쿼리에서 키워드를 추출하고 지식그래프를 탐색하여
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

        # ── Stage 1: 규칙 기반 (기존 방식, 빠름) ──
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

        return keywords[:5]  # 최대 5개

    # ── Stage 1: 규칙 기반 ──

    def _extract_by_rules(self, query: str) -> list[str]:
        """그래프 노드명 직접매칭 + 구어→의학용어 regex 패턴."""
        keywords = []

        # 노드명 직접 매칭
        for node in self.kg.graph.nodes:
            if node in query:
                keywords.append(node)

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
        regex로 못 잡는 표현도 의미적으로 매칭 가능.

        예: "밥맛이 없어" → '식욕부진' (cosine=0.52)
            "온몸이 뻣뻣해" → '뼈 통증' (cosine=0.48)
        """
        if self._node_embeddings is None:
            return []

        try:
            query_emb = self._embedding.embed([query])  # shape: (1, 384)

            # 코사인 유사도 계산
            # norm(query) * norm(nodes) → dot product / norms
            query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
            node_norms = self._node_embeddings / (
                np.linalg.norm(self._node_embeddings, axis=1, keepdims=True) + 1e-10
            )
            similarities = (query_norm @ node_norms.T).flatten()  # (num_nodes,)

            # 상위 K개 추출 + 절대 임계값 + 상대 갭 필터
            sorted_indices = np.argsort(similarities)[::-1]
            top_indices = sorted_indices[:EMBEDDING_TOP_K]

            # 상대 갭 체크: 1위 점수가 4위 점수보다 충분히 높은지
            # (갭이 작으면 모든 노드가 비슷한 점수 → 구분력 없음)
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
                    node_type = self.kg.graph.nodes[node_name].get("type", "")
                    matched.append((node_name, float(score), node_type))

            if matched:
                match_str = ", ".join(
                    f"{name}({score:.2f})" for name, score, _ in matched
                )
                logger.debug(f"🔍 임베딩 매칭: query='{query}' → {match_str}")

            return [name for name, _, _ in matched]

        except Exception as e:
            logger.debug(f"임베딩 매칭 실패: {e}")
            return []

    # ── Stage 3: LLM Fallback ──

    def _extract_by_llm(self, query: str) -> list[str]:
        """
        규칙+임베딩으로 키워드를 못 찾았을 때 LLM에게 의학 키워드 추출을 요청한다.
        동기 호출 (retrieve_node가 sync이므로).
        """
        try:
            import httpx

            # 그래프에 있는 노드 목록을 제공하여 hallucination 방지
            node_list = ", ".join(list(self.kg.graph.nodes)[:50])

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
                        "num_predict": 50,  # 키워드만이므로 짧게
                    },
                },
                timeout=5.0,  # 최대 5초 대기
            )
            response.raise_for_status()

            result = response.json().get("response", "").strip()

            # "없음" 응답 처리
            if "없음" in result or not result:
                return []

            # <think>...</think> 블록 제거
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL)
            result = re.sub(r"</?think>", "", result).strip()

            # 쉼표로 분리 후 실제 그래프 노드인지 검증
            candidates = [kw.strip() for kw in result.split(",")]
            valid_keywords = [
                kw for kw in candidates
                if kw in self.kg.graph.nodes
            ]

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
