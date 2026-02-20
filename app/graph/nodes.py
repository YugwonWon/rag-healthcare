"""
LangGraph 노드 함수들
각 노드는 ConversationState를 받아 변경할 필드만 dict로 반환한다.
"""

from app.graph import ConversationState, Intent
from app.graph.intent_classifier import classify_intent
from app.graph.query_rewriter import extract_topic
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
    """전처리: 건강 분석, 대화 이력 로딩, 반복 질문·주제 이탈 감지"""
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

    # 직전 대화에서 주제 추출 + 이전 주제 목록
    recent_topic = ""
    previous_topics: list[str] = []
    turn_count = len([h for h in conversation_history if h.get("role") == "user"])
    for entry in reversed(conversation_history):
        if entry.get("role") == "user":
            topic = extract_topic(entry["content"])
            if not recent_topic:
                recent_topic = topic
            if topic and topic not in previous_topics:
                previous_topics.append(topic)

    # ── 반복 질문 감지 ──
    repeated_question = _detect_repeated_question(message, conversation_history)

    # ── 주제 이탈 감지 ──
    topic_drifted = _detect_topic_drift(message, recent_topic, conversation_history)

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
        "previous_topics": previous_topics,
        "repeated_question": repeated_question,
        "topic_drifted": topic_drifted,
    }


async def _load_conversation_history(nickname: str) -> list[dict]:
    """대화 이력을 시간순으로 로딩한다 (최근 3턴 = 6메시지)."""
    store = get_langchain_store()
    return store.get_recent_conversations(nickname, limit=3)


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
# 노드 3: 문서 검색 (RAG + GraphRAG)
# ============================================================

def retrieve_node(state: ConversationState) -> dict:
    """벡터 검색 + 지식그래프 검색 (원본 메시지 + 건강 키워드로 검색)"""
    # 건강 키워드 확장 쿼리 또는 원본 메시지 사용 (쿼리 재작성 없음)
    query = state.get("enhanced_query") or state["message"]
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

    # GraphRAG 지식그래프 검색 (Neo4j)
    graph_context = ""
    try:
        from app.knowledge_graph import get_graph_rag
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
    """시스템 프롬프트 구성 + LLM 호출

    6가지 프롬프팅 규칙:
    1. 공감·위로 먼저
    2. 원인 설명 + 일상 대처법
    3. 유도 질문
    4. 3턴 이상 + 심각 증상 → 의료 권유
    5. 대화 마무리 금지 + 이해 확인
    6. 반복 질문 대응 + 주제 이탈 복귀
    """
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

    # ── 의도별 추가 지시 ──
    if intent == Intent.EMERGENCY:
        system_prompt += prompts.EMERGENCY_ADDENDUM
    elif intent == Intent.FOLLOWUP:
        system_prompt += prompts.FOLLOWUP_ADDENDUM
    elif intent == Intent.MEDICATION:
        system_prompt += prompts.MEDICATION_ADDENDUM

    # ── 규칙 4: 3턴 이상 + 건강 상담 → 의료 권유 ──
    turn_count = state.get("turn_count", 0)
    risk_level = state.get("risk_level", "low")
    medical_referral_given = state.get("medical_referral_given", False)

    if (
        turn_count >= 3
        and not medical_referral_given
        and intent in (Intent.HEALTH_CONSULT, Intent.MEDICATION, Intent.FOLLOWUP)
        and risk_level in ("medium", "high", "critical")
    ):
        system_prompt += prompts.MEDICAL_REFERRAL_ADDENDUM.format(turn_count=turn_count)
        medical_referral_given = True

    # ── 규칙 6: 반복 질문 감지 ──
    if state.get("repeated_question", False):
        system_prompt += prompts.REPEATED_QUESTION_ADDENDUM

    # ── 규칙 6: 주제 이탈 감지 ──
    if state.get("topic_drifted", False):
        previous_topics = state.get("previous_topics", [])
        prev_topic = previous_topics[-1] if previous_topics else "이전 주제"
        system_prompt += prompts.TOPIC_DRIFT_ADDENDUM.format(previous_topic=prev_topic)

    # ── 건강 위험 감지 시 추가 ──
    if risk_level in ("high", "critical"):
        health_analysis = state.get("health_analysis", {})
        terms = health_analysis.get("detected_health_terms", [])
        system_prompt += prompts.HIGH_RISK_ADDENDUM.format(
            risk_level=risk_level,
            risk_terms=", ".join(terms[:5]),
        )

    # LLM 호출 — 원본 메시지를 그대로 전달 (맥락 판단은 LLM이 히스토리로 수행)
    llm = get_llm()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    response = await llm.chat(messages)
    logger.info(f"💬 응답 생성 완료 | intent={intent.value} | turn={turn_count} | len={len(response)}")

    return {
        "response": response,
        "system_prompt": system_prompt,
        "medical_referral_given": medical_referral_given,
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


# ============================================================
# 반복 질문 감지 & 주제 이탈 감지
# ============================================================

def _detect_repeated_question(message: str, history: list[dict]) -> bool:
    """사용자가 이전에 했던 질문과 유사한 질문을 하고 있는지 감지한다.

    키워드 겹침 비율로 판단 (LLM 호출 없이).
    """
    if not history:
        return False

    msg_keywords = _extract_content_keywords(message)
    if not msg_keywords:
        return False

    # 과거 사용자 메시지와 비교
    for entry in history:
        if entry.get("role") != "user":
            continue
        prev_keywords = _extract_content_keywords(entry.get("content", ""))
        if not prev_keywords:
            continue
        # Overlap Coefficient: 짧은 집합 대비 겹침 비율
        intersection = msg_keywords & prev_keywords
        min_size = min(len(msg_keywords), len(prev_keywords))
        if min_size and len(intersection) / min_size >= 0.5:
            logger.info(f"🔄 반복 질문 감지 | 겹침: {intersection}")
            return True

    return False


def _detect_topic_drift(
    message: str, recent_topic: str, history: list[dict]
) -> bool:
    """현재 메시지가 최근 건강 상담 주제에서 벗어났는지 감지한다.

    건강 상담 중에만 주제 이탈을 체크한다 (일상 대화는 체크 안 함).
    """
    if not recent_topic or not history:
        return False

    # 최근 대화가 건강 관련이었는지 확인 (최근 사용자 메시지 + AI 응답)
    recent_was_health = False
    for entry in reversed(history[-4:]):
        content = entry.get("content", "")
        if any(kw in content for kw in _HEALTH_TOPIC_KEYWORDS):
            recent_was_health = True
            break

    if not recent_was_health:
        return False

    # 현재 메시지가 건강 이야기와 관련이 있는지 확인
    msg_has_health = any(kw in message for kw in _HEALTH_TOPIC_KEYWORDS)

    # 건강 상담 중이었는데 현재 메시지에 건강 키워드가 없고, 메시지가 충분히 긴 경우
    if not msg_has_health and len(message) > 10:
        logger.info(f"↩️ 주제 이탈 감지 | recent_topic={recent_topic}")
        return True

    return False


def _extract_content_keywords(text: str) -> set[str]:
    """텍스트에서 의미 있는 핵심어를 추출한다 (조사/어미 제거 → 어근 비교)."""
    import re
    words = re.findall(r"[가-힣]{2,}", text)
    # 일반적인 접속사·불용어
    stopwords = {
        "그래서", "그러면", "그런데", "하지만", "그리고", "때문에",
        "여기서", "거기서", "이것은", "그것은", "저것은", "어떻게",
        "알겠어", "네네네", "그래요",
    }
    # 한국어 조사/어미 패턴 제거 → 어근 추출
    _SUFFIX_RE = re.compile(
        r"(이요|에요|아요|어요|해요|하고|인데|은데|는데|아서|어서|해서|"
        r"이가|에서|으로|에게|까지|부터|조차|마저|이나|이랑|하면|으면|"
        r"니다|습니|세요|시다|네요|지요|거든|는지|런지|ㄴ데|ㄹ까|아도|어도|"
        r"이|을|를|은|는|에|의|과|와|도|만|로|가|요|서|고|게|며|지|니|다|자|면|죠|네|데|래|든)$"
    )
    stems = set()
    for w in words:
        if w in stopwords:
            continue
        stem = _SUFFIX_RE.sub("", w)
        if len(stem) >= 1:
            stems.add(stem)
        else:
            stems.add(w)  # 어근이 너무 짧으면 원본 유지
    return stems


# 건강 관련 주제 판별 키워드
_HEALTH_TOPIC_KEYWORDS = {
    "아프", "아파", "통증", "증상", "병원", "약", "수술", "검사",
    "진료", "치료", "건강", "질환", "질병", "피부", "혈압", "혈당",
    "당뇨", "수면", "잠", "기침", "두통", "어지러", "무릎", "허리",
    "관절", "소화", "변비", "설사", "요실금", "탈모", "갱년기",
    "난청", "골다공증", "호흡", "심장", "발톱", "손톱", "눈", "귀",
}
