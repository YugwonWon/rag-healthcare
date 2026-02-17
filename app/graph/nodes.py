"""
LangGraph 노드 함수들
각 노드는 ConversationState를 받아 변경할 필드만 dict로 반환한다.
"""

from app.graph import ConversationState, Intent
from app.graph.intent_classifier import classify_intent
from app.graph.query_rewriter import rewrite_query, extract_topic
from app.config import settings, prompts
from app.model import get_llm
from app.langchain_store import get_langchain_store
from app.preprocessing import HealthSignalDetector
from app.utils import get_kst_now, get_kst_datetime_str
from app.logger import get_logger

logger = get_logger(__name__)

# 전처리 모듈 (지연 초기화)
_health_detector: HealthSignalDetector | None = None


def _get_health_detector() -> HealthSignalDetector:
    global _health_detector
    if _health_detector is None:
        _health_detector = HealthSignalDetector(use_ner_model=True)
    return _health_detector


# ============================================================
# 노드 1: 전처리 — NER + 건강 분석 + 대화 이력 로딩
# ============================================================

async def preprocess_node(state: ConversationState) -> dict:
    """전처리: 건강 분석, 대화 이력 로딩, 시간 설정"""
    nickname = state["nickname"]
    message = state["message"]

    logger.info(f"📥 preprocess | nickname={nickname} | msg={message[:40]}...")

    # 현재 시간
    current_time = get_kst_datetime_str()

    # NER + N-gram 건강 분석
    detector = _get_health_detector()
    try:
        health_analysis = detector.get_risk_summary(message)
    except Exception as e:
        logger.warning(f"건강 분석 오류 (기본값): {e}")
        health_analysis = {
            "overall_risk": "low",
            "detected_health_terms": [],
            "risk_categories": [],
            "summary": "",
            "enhanced_query": message,
        }

    # 환자 프로필 조회
    store = get_langchain_store()
    patient_profile = await store.get_profile(nickname)

    # 대화 이력 로딩 (시간순, 최근 N개)
    conversation_history = await _load_conversation_history(nickname)

    # 직전 대화에서 주제 추출
    recent_topic = ""
    turn_count = len([h for h in conversation_history if h.get("role") == "user"])
    if conversation_history:
        for entry in reversed(conversation_history):
            if entry.get("role") == "user":
                recent_topic = extract_topic(entry["content"])
                break

    return {
        "current_time": current_time,
        "health_analysis": health_analysis,
        "enhanced_query": health_analysis.get("enhanced_query", message),
        "risk_level": health_analysis.get("overall_risk", "low"),
        "detected_symptoms": health_analysis.get("detected_health_terms", []),
        "patient_profile": patient_profile,
        "conversation_history": conversation_history,
        "recent_topic": recent_topic,
        "turn_count": turn_count,
    }


async def _load_conversation_history(nickname: str) -> list[dict]:
    """대화 이력을 시간순으로 로딩한다."""
    store = get_langchain_store()
    return store.get_recent_conversations(nickname, limit=5)


# ============================================================
# 노드 2: 의도 분류
# ============================================================

def classify_intent_node(state: ConversationState) -> dict:
    """키워드 기반 의도 분류"""
    message = state["message"]
    recent_topic = state.get("recent_topic", "")
    turn_count = state.get("turn_count", 0)

    intent, confidence = classify_intent(
        message,
        recent_topic=recent_topic,
        turn_count=turn_count,
    )

    logger.info(f"🏷️ intent={intent.value} | confidence={confidence:.2f} | msg={message[:30]}")

    # 응급 키워드 별도 추출
    from app.graph.intent_classifier import EMERGENCY_KEYWORDS
    emergency_hits = [kw for kw in EMERGENCY_KEYWORDS if kw in message]

    return {
        "intent": intent,
        "intent_confidence": confidence,
        "emergency_keywords": emergency_hits,
    }


# ============================================================
# 노드 3: 쿼리 재작성 (FOLLOWUP 의도에서만 실행)
# ============================================================

def rewrite_query_node(state: ConversationState) -> dict:
    """후속 질문일 때 이전 맥락을 반영해 쿼리를 재작성한다."""
    intent = state.get("intent")
    message = state["message"]
    history = state.get("conversation_history", [])
    recent_topic = state.get("recent_topic", "")

    if intent == Intent.FOLLOWUP:
        rewritten = rewrite_query(message, history, recent_topic)
        logger.info(f"✏️ 쿼리 재작성: '{message}' → '{rewritten}'")
        return {"rewritten_query": rewritten}

    return {"rewritten_query": message}


# ============================================================
# 노드 4: 문서 검색 (RAG + GraphRAG)
# ============================================================

def retrieve_node(state: ConversationState) -> dict:
    """벡터 검색 + 지식그래프 검색"""
    # 재작성된 쿼리 또는 확장 쿼리 사용
    query = state.get("rewritten_query") or state.get("enhanced_query") or state["message"]
    intent = state.get("intent", Intent.GENERAL_CHAT)

    # 일반 대화면 문서 검색 생략
    if intent == Intent.GENERAL_CHAT:
        return {
            "retrieved_docs": [],
            "graph_context": "",
        }

    # 벡터 검색
    retrieved_docs = []
    store = get_langchain_store()
    doc_results = store.search_documents(query, k=settings.RAG_TOP_K)
    # 대화 예제(conversations)는 제외하고 healthcare_docs만 참고 정보로 사용
    retrieved_docs = [
        d.get("content", "")[:300] for d in doc_results
        if d.get("metadata", {}).get("category") != "conversations"
    ]

    # GraphRAG 지식그래프 검색
    graph_context = ""
    try:
        from app.knowledge_graph.graph_rag import get_graph_rag
        graph_rag = get_graph_rag()
        graph_context = graph_rag.search(query)
    except Exception as e:
        logger.debug(f"GraphRAG 검색 스킵: {e}")

    logger.info(f"🔍 검색 완료 | docs={len(retrieved_docs)} | graph={'있음' if graph_context else '없음'}")

    return {
        "retrieved_docs": retrieved_docs,
        "graph_context": graph_context,
    }


# ============================================================
# 노드 5-A: 응급 상황 처리
# ============================================================

def emergency_node(state: ConversationState) -> dict:
    """응급 상황 알림 생성"""
    nickname = state["nickname"]
    message = state["message"]
    keywords = state.get("emergency_keywords", [])
    profile = state.get("patient_profile", {})

    emergency_contact = profile.get("emergency_contact", "")

    alert = {
        "level": "critical",
        "message": message,
        "keywords": keywords,
        "nickname": nickname,
        "emergency_contact": emergency_contact,
        "action_required": True,
    }

    logger.warning(f"🚨 응급 알림 생성 | nickname={nickname} | keywords={keywords}")

    return {"emergency_alert": alert}


# ============================================================
# 노드 6: LLM 응답 생성
# ============================================================

async def generate_response_node(state: ConversationState) -> dict:
    """시스템 프롬프트 구성 + LLM 호출"""
    intent = state.get("intent", Intent.GENERAL_CHAT)
    message = state["message"]
    nickname = state["nickname"]

    # 프롬프트 구성 요소
    current_time = state.get("current_time", "")
    patient_info = _format_patient_info(state.get("patient_profile"))
    conversation_history = _format_history(state.get("conversation_history", []))
    retrieved_context = _format_docs(state.get("retrieved_docs", []))
    graph_context = state.get("graph_context", "")

    # 일반 대화면 참고 정보를 비워서 자연스러운 대화 유도
    if intent == Intent.GENERAL_CHAT:
        retrieved_context = "일반 대화 - 참고 정보 불필요"
        graph_context = ""

    # 그래프 컨텍스트를 참고 정보에 추가
    if graph_context:
        retrieved_context = f"{retrieved_context}\n\n[건강 지식그래프]\n{graph_context}"

    # 시스템 프롬프트 구성
    system_prompt = prompts.SYSTEM_PROMPT.format(
        current_time=current_time,
        patient_info=patient_info,
        conversation_history=conversation_history,
        retrieved_context=retrieved_context,
    )

    # 의도별 추가 지시
    if intent == Intent.EMERGENCY:
        system_prompt += (
            "\n\n[⚠️ 응급 상황 감지]\n"
            "사용자에게 즉시 119에 전화하거나 보호자에게 연락하도록 안내하세요.\n"
            "침착하게 현재 상태를 확인하고, 안전한 자세를 유지하도록 안내합니다."
        )
    elif intent == Intent.FOLLOWUP:
        rewritten = state.get("rewritten_query", message)
        system_prompt += f"\n\n[맥락 참고] 사용자의 질문은 이전 대화의 후속입니다. 재작성된 질문: {rewritten}"
    elif intent == Intent.MEDICATION:
        system_prompt += "\n\n[복약 관련] 약에 대한 정보를 정확하게 안내하되, 반드시 의사나 약사와 상담하도록 권유하세요."

    # 건강 위험 감지 시 추가
    risk_level = state.get("risk_level", "low")
    if risk_level in ("high", "critical"):
        health_analysis = state.get("health_analysis", {})
        terms = health_analysis.get("detected_health_terms", [])
        system_prompt += f"\n\n[건강 위험 감지] 위험 수준: {risk_level}, 감지 용어: {', '.join(terms[:5])}"
        system_prompt += "\n주의: 필요시 보호자 연락이나 전문가 상담을 안내하세요."

    # LLM 호출
    llm = get_llm()
    user_msg = state.get("rewritten_query", message) if intent == Intent.FOLLOWUP else message

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    response = await llm.chat(messages)
    logger.info(f"💬 응답 생성 완료 | intent={intent.value} | len={len(response)}")

    return {
        "response": response,
        "system_prompt": system_prompt,
    }


# ============================================================
# 노드 7: 대화 저장
# ============================================================

def save_conversation_node(state: ConversationState) -> dict:
    """대화 기록을 저장한다."""
    nickname = state["nickname"]
    message = state["message"]
    response = state.get("response", "")
    health_analysis = state.get("health_analysis", {})
    intent = state.get("intent", Intent.GENERAL_CHAT)

    metadata = {
        "intent": intent.value if isinstance(intent, Intent) else str(intent),
        "health_terms": health_analysis.get("detected_health_terms", [])[:5],
        "risk_level": state.get("risk_level", "low"),
        "risk_categories": [
            r["category"] for r in health_analysis.get("risk_categories", [])
        ],
    }

    try:
        store = get_langchain_store()
        store.save_conversation(nickname, message, response)
        logger.debug(f"💾 대화 저장 완료 | nickname={nickname}")
    except Exception as e:
        logger.error(f"대화 저장 오류: {e}")

    return {}


# ============================================================
# 헬퍼 함수
# ============================================================

def _format_patient_info(profile: dict | None) -> str:
    if not profile:
        return "등록된 환자 정보가 없습니다."
    lines = []
    for k, v in profile.items():
        if k != "nickname" and v:
            lines.append(f"- {k}: {v}")
    return "\n".join(lines) if lines else "기본 정보만 등록됨"


def _format_history(history: list[dict]) -> str:
    if not history:
        return "이전 대화 없음"
    parts = []
    for entry in history[-6:]:  # 최근 3쌍
        role = "사용자" if entry.get("role") == "user" else "AI"
        content = entry.get("content", "")[:200]
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _format_docs(docs: list[str]) -> str:
    if not docs:
        return "관련 의료 정보 없음"
    parts = []
    for i, doc in enumerate(docs[:3], 1):
        parts.append(f"[{i}] {doc}")
    return "\n\n".join(parts)
