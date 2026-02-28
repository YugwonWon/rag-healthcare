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
        _health_detector = HealthSignalDetector()
    return _health_detector


def _has_health_signal(message: str) -> bool:
    """빠른 키워드 사전 필터: 건강 관련 키워드가 있으면 True → 형태소 분석 실행."""
    return any(kw in message for kw in _HEALTH_TOPIC_KEYWORDS)


def _default_health_analysis(message: str) -> dict:
    """형태소 분석 스킵 시 기본 건강 분석 결과."""
    return {
        "overall_risk": "low",
        "detected_health_terms": [],
        "risk_categories": [],
        "summary": "",
        "enhanced_query": message,
    }


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

    # 형태소 분석 + N-gram 건강 분석 (건강 키워드가 없으면 스킵하여 속도 개선)
    if _has_health_signal(message):
        detector = _get_health_detector()
        try:
            health_analysis = detector.get_risk_summary(message)
        except Exception as e:
            logger.warning(f"건강 분석 오류 (기본값): {e}")
            health_analysis = _default_health_analysis(message)
    else:
        logger.debug(f"건강 키워드 없음 → 분석 스킵 | msg={message[:30]}")
        health_analysis = _default_health_analysis(message)

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
    """인텐트별 프롬프트 선택 + LLM 호출

    2.1B 소형 모델에 맞게 인텐트별로 분리된 짧은 프롬프트를 사용한다.
    음성 기반 챗봇에 적합한 1~3문장 응답을 유도한다.
    """
    intent = state.get("intent", Intent.GENERAL_CHAT)
    message = state["message"]
    nickname = state["nickname"]

    # 공통 컨텍스트 변수
    current_time = state.get("current_time", "")
    patient_info = _format_patient_info(state.get("patient_profile"))
    conversation_history = _format_history(state.get("conversation_history", []))
    retrieved_context = _format_docs(state.get("retrieved_docs", []))
    graph_context = state.get("graph_context", "")

    # 그래프 컨텍스트를 참고 정보에 추가
    if graph_context:
        retrieved_context = f"{retrieved_context}\n\n[건강 지식그래프]\n{graph_context}"

    # ── 인텐트별 프롬프트 선택 ──
    fmt_kwargs = dict(
        current_time=current_time,
        patient_info=patient_info,
        conversation_history=conversation_history,
        retrieved_context=retrieved_context,
    )

    prompt_map = {
        Intent.GENERAL_CHAT: prompts.GENERAL_CHAT_PROMPT,
        Intent.HEALTH_CONSULT: prompts.HEALTH_CONSULT_PROMPT,
        Intent.EMERGENCY: prompts.EMERGENCY_PROMPT,
        Intent.FOLLOWUP: prompts.FOLLOWUP_PROMPT,
        Intent.MEDICATION: prompts.MEDICATION_PROMPT,
        Intent.LIFESTYLE: prompts.LIFESTYLE_PROMPT,
    }
    template = prompt_map.get(intent, prompts.GENERAL_CHAT_PROMPT)

    # retrieved_context가 없는 프롬프트(GENERAL_CHAT, EMERGENCY)는 해당 키 제거
    if "{retrieved_context}" not in template:
        fmt_kwargs.pop("retrieved_context", None)

    system_prompt = template.format(**fmt_kwargs)

    # ── 조건부 addendum (건강 상담 계열만) ──
    turn_count = state.get("turn_count", 0)
    risk_level = state.get("risk_level", "low")
    medical_referral_given = state.get("medical_referral_given", False)

    health_intents = (Intent.HEALTH_CONSULT, Intent.MEDICATION, Intent.FOLLOWUP)

    if (
        turn_count >= 3
        and not medical_referral_given
        and intent in health_intents
        and risk_level in ("medium", "high", "critical")
    ):
        system_prompt += prompts.MEDICAL_REFERRAL_ADDENDUM
        medical_referral_given = True

    if state.get("repeated_question", False) and intent in health_intents:
        system_prompt += prompts.REPEATED_QUESTION_ADDENDUM

    if state.get("topic_drifted", False) and intent in health_intents:
        previous_topics = state.get("previous_topics", [])
        prev_topic = previous_topics[-1] if previous_topics else "이전 주제"
        system_prompt += prompts.TOPIC_DRIFT_ADDENDUM.format(previous_topic=prev_topic)

    if risk_level in ("high", "critical"):
        health_analysis = state.get("health_analysis", {})
        terms = health_analysis.get("detected_health_terms", [])
        system_prompt += prompts.HIGH_RISK_ADDENDUM.format(
            risk_level=risk_level,
            risk_terms=", ".join(terms[:5]),
        )

    # LLM 호출
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

    건강 관련 키워드가 포함된 질문만 대상으로 하며,
    겹침 비율(Overlap Coefficient ≥ 0.7)과 최소 겹침 키워드 3개를 요구한다.
    """
    if not history:
        return False

    # 건강 키워드가 없는 일상 대화는 반복 감지 대상 아님
    if not any(kw in message for kw in _HEALTH_TOPIC_KEYWORDS):
        return False

    msg_keywords = _extract_content_keywords(message)
    if len(msg_keywords) < 3:  # 키워드 2개 이하면 반복 감지 불가
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
        if min_size >= 3 and len(intersection) >= 3 and len(intersection) / min_size >= 0.7:
            logger.info(f"🔄 반복 질문 감지 | 겹침: {intersection}")
            return True

    return False


def _detect_topic_drift(
    message: str, recent_topic: str, history: list[dict]
) -> bool:
    """현재 메시지가 최근 건강 상담 주제에서 벗어났는지 감지한다.

    최근 '사용자 발화'에 건강 키워드가 2개 이상 있었을 때만
    주제 이탈로 판단한다 (AI 응답은 기준에서 제외).
    일상→일상 전환은 주제 이탈로 보지 않는다.
    """
    if not recent_topic or not history:
        return False

    # 최근 **사용자** 메시지가 건강 관련이었는지 확인 (AI 응답은 제외)
    recent_user_health_count = 0
    for entry in reversed(history[-4:]):
        if entry.get("role") != "user":
            continue
        content = entry.get("content", "")
        hits = sum(1 for kw in _HEALTH_TOPIC_KEYWORDS if kw in content)
        recent_user_health_count = max(recent_user_health_count, hits)

    # 최근 사용자 발화에 건강 키워드가 2개 미만이면 건강 상담이 아니었음
    if recent_user_health_count < 2:
        return False

    # 현재 메시지가 건강 이야기와 관련이 있는지 확인
    msg_has_health = any(kw in message for kw in _HEALTH_TOPIC_KEYWORDS)

    # 건강 상담 중이었는데 현재 메시지에 건강 키워드가 없고, 메시지가 충분히 긴 경우
    if not msg_has_health and len(message) > 15:
        logger.info(f"↩️ 주제 이탈 감지 | recent_topic={recent_topic}")
        return True

    return False


def _extract_content_keywords(text: str) -> set[str]:
    """텍스트에서 의미 있는 핵심어를 추출한다 (조사/어미 제거 → 어근 비교)."""
    import re
    words = re.findall(r"[가-힣]{2,}", text)
    # 일반적인 접속사·불용어·일상 동사/부사
    stopwords = {
        # 접속사·지시사
        "그래서", "그러면", "그런데", "하지만", "그리고", "때문에",
        "여기서", "거기서", "이것은", "그것은", "저것은", "어떻게",
        "알겠어", "네네네", "그래요", "그래도", "그러니", "그러나",
        # 일상 동사/부사 (반복 감지 false positive 방지)
        "이제", "지금", "나갈", "나가", "가려", "하려", "하고",
        "먹었", "먹어", "했어", "했다", "할거",
        "좋은", "좋아", "잘되",
        "너무", "정말", "조금", "그냥", "많이",
        "오늘", "내일", "어제", "아까", "요즘", "나중",
        # 일상 대화에서 흔한 동사/명사 어근 (false positive 방지)
        "얘기", "이야기", "재밌", "재미", "심심", "하나",
        "뭐가", "뭐라", "뭐든", "무슨", "어떤", "왜냐",
        "해줘", "해주", "알려", "들려", "하자", "할까",
        "같이", "혼자", "우리", "나는", "저는", "거기", "여기",
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
        if len(stem) >= 2:  # 어근 2자 이상만 유의미
            stems.add(stem)
    return stems


# 건강 관련 주제 판별 키워드 (실제 건강 불만/증상 표현만)
_HEALTH_TOPIC_KEYWORDS = {
    "아프", "아파", "통증", "증상", "병원", "약", "수술", "검사",
    "진료", "치료", "질환", "질병", "피부", "혆압", "혆당",
    "당뇨", "수면", "잠", "기침", "두통", "어지러", "무릎", "허리",
    "관절", "변비", "설사", "요실금", "탈모", "갱년기",
    "난청", "골다공증", "호흡", "심장", "발톱", "손톱", "눈", "귀",
    "소화불량", "소화가 안",
}
