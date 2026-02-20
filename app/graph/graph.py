"""
LangGraph 대화 그래프 정의
의도에 따라 분기하는 상태 머신 그래프

흐름:
  START → preprocess → classify_intent → [분기]
    → GENERAL_CHAT     → generate_response → save → END
    → HEALTH_CONSULT   → retrieve → generate_response → save → END
    → FOLLOWUP         → retrieve → generate_response → save → END
    → EMERGENCY        → emergency_alert → retrieve → generate_response → save → END
    → MEDICATION       → retrieve → generate_response → save → END
    → LIFESTYLE        → retrieve → generate_response → save → END

쿼리 재작성 없이 원본 메시지로 검색하고,
대화 히스토리는 LLM 프롬프트에 직접 주입하여 맥락을 판단한다.
"""

from langgraph.graph import StateGraph, END

from app.graph import ConversationState, Intent
from app.graph.nodes import (
    preprocess_node,
    classify_intent_node,
    retrieve_node,
    emergency_node,
    generate_response_node,
    save_conversation_node,
)
from app.logger import get_logger

logger = get_logger(__name__)


def _route_by_intent(state: ConversationState) -> str:
    """의도에 따라 다음 노드를 결정한다."""
    intent = state.get("intent", Intent.GENERAL_CHAT)

    if intent == Intent.EMERGENCY:
        return "emergency"
    elif intent == Intent.GENERAL_CHAT:
        return "general"
    else:
        # HEALTH_CONSULT, FOLLOWUP, MEDICATION, LIFESTYLE
        # 모두 검색이 필요한 경로
        return "needs_retrieval"


def build_conversation_graph() -> StateGraph:
    """대화 처리 그래프를 구성하고 컴파일한다."""

    graph = StateGraph(ConversationState)

    # ── 노드 등록 ──
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("emergency", emergency_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("save_conversation", save_conversation_node)

    # ── 엣지 구성 ──

    # START → preprocess → classify_intent
    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "classify_intent")

    # classify_intent → 조건부 분기
    graph.add_conditional_edges(
        "classify_intent",
        _route_by_intent,
        {
            "emergency": "emergency",
            "general": "generate_response",
            "needs_retrieval": "retrieve",
        },
    )

    # emergency → retrieve → generate_response
    graph.add_edge("emergency", "retrieve")
    graph.add_edge("retrieve", "generate_response")

    # generate_response → save_conversation → END
    graph.add_edge("generate_response", "save_conversation")
    graph.add_edge("save_conversation", END)

    # ── 컴파일 ──
    compiled = graph.compile()
    logger.info("✅ LangGraph 대화 그래프 컴파일 완료")

    return compiled


# 싱글톤 컴파일된 그래프
_compiled_graph = None


def get_conversation_graph():
    """컴파일된 그래프 싱글톤 반환"""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_conversation_graph()
    return _compiled_graph
