"""
키워드 기반 의도 분류기
LLM 호출 없이 규칙 기반으로 의도를 분류하여 응답 속도 유지

1.2B 소형 모델 환경에서 추가 LLM 호출을 최소화하기 위해
키워드 매칭 + 패턴 기반 분류를 사용한다.
"""

import re
from typing import Tuple

from app.graph import Intent
from app.logger import get_logger

logger = get_logger(__name__)


# ── 프롬프트 유출 시도 키워드 ──
PROMPT_LEAK_KEYWORDS = [
    "시스템 프롬프트", "system prompt", "프롬프트를 알려", "프롬프트 알려",
    "지시를 무시", "ignore previous", "ignore instructions",
    "당신의 지시", "너의 지시", "instructions are",
    "프롬프트 내용", "초기 지시", "설정을 알려", "역할을 알려",
    "jailbreak", "jail break", "역할극", "roleplay as",
    "pretend you are", "act as if", "act as a",
    "이전 지시를 무시", "규칙을 무시", "규칙을 어겨",
]

# ── 응급 키워드 (최우선 매칭) ──
EMERGENCY_KEYWORDS = [
    "쓰러", "의식", "호흡", "숨을 못", "숨이 안",
    "가슴이 아", "가슴 통증", "심장", "경련", "발작",
    "피를 토", "피가 나", "출혈", "골절", "부러",
    "119", "응급", "구급", "살려", "죽",
    "머리를 부딪", "머리를 다쳤", "머리에서 피",
    "화상", "불에 데", "뜨거운 물",
    "약을 너무 많이", "약 과다", "중독",
    "못 움직", "마비", "감각이 없",
    "갑자기 말을 못", "얼굴이 돌아",
]

# ── 건강 상담 키워드 ──
HEALTH_KEYWORDS = [
    # 증상
    "아프", "아파", "통증", "쑤시", "저리", "가려", "붓",
    "두통", "어지러", "메스꺼", "구역질", "설사", "변비",
    "소화가 안", "소화불량", "소화 안 되", "소화가 잘 안",
    "열이", "기침", "콧물", "재채기", "감기", "몸살",
    "숨이 차", "가래", "목이 아",
    # 수면
    "잠을 못", "잠이 안", "불면", "수면", "밤에 깨",
    "새벽에 깨", "잠을 설", "꿈을 많이",
    # 질환/상태
    "당뇨", "혈당", "혈압", "고혈압", "저혈압",
    "관절", "뼈", "근육", "허리", "무릎", "어깨",
    "눈이 침침", "노안", "백내장", "녹내장",
    "귀가 안", "난청", "이명",
    "치매", "건망증", "기억",
    "발톱", "손톱", "피부", "탈모", "구강", "잇몸", "치아",
    "갱년기", "골다공증", "요실금", "변실금",
    "욕창", "구축", "폐", "폐렴",
    # 문의
    "병원", "진료", "검사", "진단", "처방", "치료",
    "증상이", "증세", "상태가",
]

# ── 복약 키워드 ──
MEDICATION_KEYWORDS = [
    "약", "복약", "복용", "처방전", "먹는 약",
    "부작용", "약 이름", "약 먹", "약을 안",
    "언제 먹", "몇 알", "식전", "식후",
]

# ── 생활습관 키워드 ──
LIFESTYLE_KEYWORDS = [
    "밥", "식사", "식단", "음식", "영양", "비타민",
    "운동", "산책", "걷기", "체조", "스트레칭",
    "목욕", "씻", "위생",
    "TV", "취미", "여가", "산책",
]

# ── 후속 질문 패턴 (맥락 의존) ──
FOLLOWUP_PATTERNS = [
    r"^(그래서|그러면|그럼|그게|그건)[\s?]",
    r"^(어떻게|어떡|뭘|무엇을|뭘|왜|언제|얼마나)[\s?]",
    r"^(더|또|다른|추가로|그리고)[\s]",
    r"^(그거|그것|이거|저거|그런|이런|저런)\s",
    r"^(해야|하면|되나|될까|좋을까|좋아|나아|괜찮)",
    r"^(네|응|예|맞아|알겠|그래)\s",
    r"^.{1,8}(해야 해|하면 돼|좋아\?|될까|인가요|인가|건가요|건가|할까)",
    r"^.{1,6}\?$",  # 짧은 질문 (6글자 이하 + 물음표)
]

# ── 일반 대화 패턴 ──
GENERAL_CHAT_PATTERNS = [
    r"^(안녕|반갑|감사|고마|잘 지내|오래간만|오랜만)",
    r"^(좋은 아침|좋은 하루|잘 자|잘 잤)",
    r"(날씨|오늘|내일|어제|주말|휴일)",
    r"^(심심|외로|그리|보고 싶|울적|기분)",
    r"^(뭐 해|뭐 하|뭐해|뭐하)",
    r"(먹었|먹어|내가 해|나갈 거|가려고|다녀왔|했어)",
    r"(비빔밥|밥 먹|점심|저녁|아침 먹|배 부르|배불러)",
    r"(산책 나갈|산책 가|산책하|운동하러|걸을 거|나갈 거)",
]


def classify_intent(
    message: str,
    recent_topic: str = "",
    turn_count: int = 0,
) -> Tuple[Intent, float]:
    """
    사용자 메시지의 의도를 분류한다.

    Returns:
        (Intent, confidence)  confidence는 0.0~1.0
    """
    msg = message.strip()
    msg_lower = msg.lower()

    # ── 0단계: 프롬프트 유출 시도 차단 ──
    if any(kw in msg_lower for kw in PROMPT_LEAK_KEYWORDS):
        logger.warning(f"🚫 프롬프트 유출 시도 감지: {msg[:50]}")
        return Intent.BLOCKED, 1.0

    # ── 1단계: 응급 상황 체크 (최우선) ──
    emergency_hits = [kw for kw in EMERGENCY_KEYWORDS if kw in msg_lower]
    if emergency_hits:
        logger.warning(f"🚨 응급 키워드 감지: {emergency_hits}")
        return Intent.EMERGENCY, 0.95

    # ── 2단계: 일반 대화 체크 (인사/감정 — followup보다 우선) ──
    for pattern in GENERAL_CHAT_PATTERNS:
        if re.search(pattern, msg):
            return Intent.GENERAL_CHAT, 0.8

    # ── 3단계: 후속 질문 판별 ──
    # 짧은 메시지(15자 이하) + 이전 대화가 있으면 후속 질문 가능성 높음
    if turn_count > 0 and recent_topic:
        for pattern in FOLLOWUP_PATTERNS:
            if re.search(pattern, msg):
                logger.debug(f"후속 질문 패턴 매칭: {pattern}")
                return Intent.FOLLOWUP, 0.85

        # 매우 짧은 메시지는 후속 질문으로 간주
        if len(msg) <= 10:
            return Intent.FOLLOWUP, 0.75

    # ── 4단계: 건강 상담 체크 ──
    health_hits = [kw for kw in HEALTH_KEYWORDS if kw in msg_lower]
    if len(health_hits) >= 2:
        return Intent.HEALTH_CONSULT, 0.9
    if len(health_hits) == 1:
        return Intent.HEALTH_CONSULT, 0.75

    # ── 5단계: 복약 체크 ──
    med_hits = [kw for kw in MEDICATION_KEYWORDS if kw in msg_lower]
    if med_hits:
        return Intent.MEDICATION, 0.8

    # ── 6단계: 생활습관 체크 ──
    life_hits = [kw for kw in LIFESTYLE_KEYWORDS if kw in msg_lower]
    if len(life_hits) >= 2:
        return Intent.LIFESTYLE, 0.8
    if len(life_hits) == 1:
        return Intent.LIFESTYLE, 0.65

    # ── 기본값: 메시지 길이 기반 추정 ──
    if len(msg) <= 10 and turn_count > 0:
        return Intent.FOLLOWUP, 0.6

    return Intent.GENERAL_CHAT, 0.5
