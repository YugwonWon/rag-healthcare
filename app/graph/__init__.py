"""
LangGraph 대화 상태 정의
치매노인 헬스케어 챗봇의 멀티턴 대화 상태 관리
"""

from typing import TypedDict, Optional, Annotated
from enum import Enum


class Intent(str, Enum):
    """사용자 의도 분류"""
    GENERAL_CHAT = "general_chat"          # 일반 대화 (안부, 잡담)
    HEALTH_CONSULT = "health_consult"      # 건강 상담 (증상, 질환 문의)
    EMERGENCY = "emergency"                # 위급 상황 (응급 증상)
    FOLLOWUP = "followup"                  # 후속 질문 (이전 맥락 의존)
    MEDICATION = "medication"              # 복약 관련
    LIFESTYLE = "lifestyle"               # 생활습관 (식사, 운동, 수면)


class ConversationState(TypedDict, total=False):
    """LangGraph 대화 상태

    그래프의 모든 노드가 공유하는 상태 객체.
    각 노드는 필요한 필드만 읽고, 변경된 필드만 반환한다.
    """

    # ── 입력 ──
    nickname: str                          # 사용자 닉네임
    message: str                           # 원본 사용자 메시지

    # ── 의도 분류 ──
    intent: Intent                         # 분류된 의도
    intent_confidence: float               # 분류 신뢰도 (0~1)

    # ── 맥락 & 이력 ──
    conversation_history: list[dict]       # 최근 대화 이력 [{role, content}, ...]
    recent_topic: str                      # 직전 대화 주제 (멀티턴 맥락)
    turn_count: int                        # 현재 세션 대화 턴 수

    # ── 쿼리 처리 ──
    rewritten_query: str                   # 재작성된 쿼리 (맥락 반영)
    enhanced_query: str                    # NER/건강 용어 확장 쿼리

    # ── 검색 결과 ──
    retrieved_docs: list[str]              # 벡터 검색 결과 문서들
    graph_context: str                     # GraphRAG 지식그래프 컨텍스트
    patient_profile: dict                  # 환자 프로필

    # ── 건강 분석 ──
    health_analysis: dict                  # NER + N-gram 건강 분석 결과
    risk_level: str                        # 위험 수준: low/medium/high/critical
    detected_symptoms: list[str]           # 감지된 증상들
    emergency_keywords: list[str]          # 감지된 응급 키워드

    # ── 응답 ──
    response: str                          # 최종 AI 응답
    system_prompt: str                     # 구성된 시스템 프롬프트
    emergency_alert: dict                  # 위급 상황 알림 정보

    # ── 메타데이터 ──
    current_time: str                      # 현재 한국 시간
    error: Optional[str]                   # 에러 메시지 (있을 경우)
