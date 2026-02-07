"""
쿼리 재작성 모듈
후속 질문(FOLLOWUP)에서 이전 대화 맥락을 반영하여 쿼리를 재작성한다.

LLM 호출을 최소화하기 위해 규칙 기반 재작성을 먼저 시도하고,
불가능한 경우에만 LLM을 사용한다.
"""

import re
from typing import Optional

from app.logger import get_logger

logger = get_logger(__name__)


def rewrite_query(
    current_message: str,
    conversation_history: list[dict],
    recent_topic: str = "",
) -> str:
    """
    후속 질문을 이전 맥락을 포함한 독립적 쿼리로 재작성한다.

    예시:
        이전: "발톱이 안으로 막 들어가"
        현재: "어떻게 해야해?"
        결과: "발톱이 안으로 들어갈 때 어떻게 해야 하나요?"

    Args:
        current_message: 현재 사용자 메시지
        conversation_history: 최근 대화 이력 [{role, content}, ...]
        recent_topic: 직전 대화 주제

    Returns:
        재작성된 쿼리 문자열
    """
    msg = current_message.strip()

    # 이전 대화가 없으면 원본 반환
    if not conversation_history:
        return msg

    # 마지막 사용자 메시지와 AI 응답 추출
    last_user_msg = ""
    last_ai_msg = ""
    for entry in reversed(conversation_history):
        if entry.get("role") == "assistant" and not last_ai_msg:
            last_ai_msg = entry.get("content", "")
        elif entry.get("role") == "user" and not last_user_msg:
            last_user_msg = entry.get("content", "")
        if last_user_msg and last_ai_msg:
            break

    if not last_user_msg:
        return msg

    # ── 규칙 기반 재작성 패턴 ──

    # 패턴 1: "어떻게 해야 해?" 계열
    if re.search(r"(어떻게|어떡|어째)", msg):
        rewritten = f"{last_user_msg} {msg}"
        logger.info(f"쿼리 재작성 (어떻게): '{msg}' → '{rewritten}'")
        return rewritten

    # 패턴 2: "그래서/그러면" 계열 (결과/대안 질문)
    match = re.match(r"^(그래서|그러면|그럼)\s*(.*)$", msg)
    if match:
        continuation = match.group(2) or "어떻게 하면 되나요?"
        rewritten = f"{last_user_msg} 관련해서 {continuation}"
        logger.info(f"쿼리 재작성 (그래서): '{msg}' → '{rewritten}'")
        return rewritten

    # 패턴 3: "더/또/다른" 계열 (추가 정보)
    match = re.match(r"^(더|또|다른|추가로|그리고)\s*(.*)$", msg)
    if match:
        additional = match.group(2) or "알려주세요"
        rewritten = f"{last_user_msg}에 대해 {additional}"
        logger.info(f"쿼리 재작성 (추가): '{msg}' → '{rewritten}'")
        return rewritten

    # 패턴 4: 짧은 응답/확인 ("네", "맞아", "그래")
    if re.match(r"^(네|응|예|맞아|알겠|그래|좋아)\s*(.*)$", msg):
        # 이전 AI 응답에서 제안사항이 있었으면 그걸 이어감
        return f"{last_user_msg}에 대해 더 알려주세요"

    # 패턴 5: 대명사 참조 ("그거", "그것", "이거")
    match = re.match(r"^(그거|그것|그건|이거|저거|그런|이런)\s*(.*)$", msg)
    if match:
        rest = match.group(2) or ""
        topic = recent_topic or last_user_msg
        rewritten = f"{topic} {rest}".strip()
        logger.info(f"쿼리 재작성 (대명사): '{msg}' → '{rewritten}'")
        return rewritten

    # 패턴 6: 매우 짧은 질문 (10자 이하)
    if len(msg) <= 10:
        rewritten = f"{last_user_msg} {msg}"
        logger.info(f"쿼리 재작성 (짧은 질문): '{msg}' → '{rewritten}'")
        return rewritten

    # 재작성 불필요 — 충분히 독립적인 질문
    return msg


def extract_topic(message: str) -> str:
    """
    메시지에서 핵심 주제를 추출한다.
    대화 맥락 유지를 위해 recent_topic으로 저장한다.

    Args:
        message: 사용자 메시지

    Returns:
        추출된 주제 문자열
    """
    # 조사/어미 제거하여 핵심어 추출
    # 예: "발톱이 안으로 막 들어가" → "발톱 안쪽 들어감"
    topic_keywords = []

    # 건강 관련 명사 패턴
    health_nouns = re.findall(
        r"(발톱|손톱|피부|머리|눈|귀|코|입|목|허리|무릎|어깨|손|발|"
        r"가슴|배|등|팔|다리|관절|근육|뼈|잇몸|치아|"
        r"혈압|혈당|당뇨|수면|잠|약|통증|두통|기침|"
        r"구강|탈모|갱년기|요실금|욕창|변비|설사|소화|"
        r"폐|심장|간|신장|위|장|대장|소장|방광|"
        r"난청|노안|골다공증|구축|식욕)",
        message
    )
    topic_keywords.extend(health_nouns)

    # 상태/증상 동사 패턴
    symptom_verbs = re.findall(
        r"(아프|아파|저리|가려|붓|쑤시|안 나|안나|못 자|못자|"
        r"안 들|안들|들어가|빠지|떨리|어지러|메스꺼)",
        message
    )
    topic_keywords.extend(symptom_verbs)

    if topic_keywords:
        return " ".join(dict.fromkeys(topic_keywords))  # 중복 제거 유지 순서

    # 핵심어가 없으면 메시지 앞 20자
    return message[:20].strip()
