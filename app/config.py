"""
환경설정 관리 모듈 (pydantic-settings)
치매노인 헬스케어 RAG 챗봇 설정
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # 기본 설정
    APP_NAME: str = "치매노인 맞춤형 헬스케어 RAG 챗봇"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    
    # API 설정
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # LLM 설정 (Ollama)
    # ============================================================
    # 지원 모델 목록:
    # - exaone-counseling: LGAI-EXAONE/EXAONE-4.0-1.2B (이전 모델)
    #   └ 특징: 1.28B 경량, 온디바이스 최적화, 한국어 지원
    #   └ GGUF: models/EXAONE-4.0-1.2B-Q4_K_M.gguf
    #   └ CPU 추론 속도 우수 (1.2B 경량)
    #
    # - kanana-counseling: Kakao Kanana-nano 2.1B Instruct (현재 모델)
    #   └ HuggingFace: kakaocorp/kanana-nano-2.1b-instruct
    #   └ GGUF: kanana-nano-2.1b-instruct-q4_k_m.gguf
    #
    # - qwen2.5:3b: Alibaba Qwen 2.5 3B
    #   └ Ollama 공식 모델
    # ============================================================
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "kanana-counseling"  # Kanana 2.1B 파인튜닝된 상담 모델
    LLM_TEMPERATURE: float = 0.4  # 다양한 유도 질문 생성을 위해 0.1 → 0.4
    LLM_MAX_TOKENS: int = 512  # 공감+설명+유도질문 공간 확보 (256은 유도질문 전에 잘림)
    
    # OpenAI 설정 (Fallback용)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    # 임베딩 설정 (온디바이스)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_DEVICE: str = "cpu"  # cpu, cuda, mps
    
    # PostgreSQL + pgvector 설정 (LangChain 데이터 레이어)
    # Cloud SQL 연결 문자열 형식:
    # - 로컬: postgresql://user:pass@localhost:5432/dbname
    # - Cloud SQL (Unix Socket): postgresql://user:pass@/dbname?host=/cloudsql/project:region:instance
    DATABASE_URL: Optional[str] = None
    
    # Cloud SQL 개별 환경 변수 (Secret Manager 지원)
    DB_HOST: Optional[str] = None  # /cloudsql/project:region:instance
    DB_NAME: Optional[str] = None
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None  # Secret Manager에서 주입
    
    @property
    def database_url(self) -> Optional[str]:
        """DATABASE_URL 또는 개별 환경 변수로 연결 문자열 생성"""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        # 개별 환경 변수에서 조합 (Cloud Run + Secret Manager)
        if all([self.DB_HOST, self.DB_NAME, self.DB_USER, self.DB_PASSWORD]):
            if self.DB_HOST.startswith("/cloudsql/"):
                # Unix 소켓 연결
                return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@/{self.DB_NAME}?host={self.DB_HOST}"
            else:
                # TCP 연결
                return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}/{self.DB_NAME}"
        return None

    
    # 대화 기록 설정
    CONVERSATION_COLLECTION_NAME: str = "conversations"
    MAX_CONVERSATION_HISTORY: int = 10
    
    # 대화 요약 설정
    CONVERSATION_SUMMARY_THRESHOLD: int = 10  # 요약 생성 기준 대화 수
    CONVERSATION_SUMMARY_ENABLED: bool = True  # 요약 기능 활성화 여부
    
    # RAG 설정
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.5
    
    # GraphRAG 설정
    GRAPHRAG_ENABLED: bool = True  # GraphRAG 지식그래프 사용 여부
    GRAPHRAG_MAX_DEPTH: int = 2    # 지식그래프 탐색 깊이
    
    # Neo4j 설정
    NEO4J_URI: Optional[str] = None
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: Optional[str] = None
    NEO4J_DATABASE: str = "neo4j"
    
    # LangGraph 설정
    LANGGRAPH_FOLLOWUP_MIN_CONFIDENCE: float = 0.5  # 후속 질문 분류 최소 신뢰도
    
    # 헬스케어 도메인 설정
    PATIENT_PROFILE_COLLECTION: str = "patient_profiles"
    MEDICATION_REMINDER_ENABLED: bool = True
    DAILY_ROUTINE_TRACKING: bool = False  # 일상 루틴 추적 비활성화 (대화 맥락 유지)
    
    # 파인튜닝 모델 설정
    FINETUNED_MODEL_PATH: Optional[str] = None
    USE_FINETUNED_MODEL: bool = False
    
    # Cloud Run 설정
    CLOUD_RUN_URL: Optional[str] = None
    
    # HuggingFace 설정
    HF_TOKEN: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # 정의되지 않은 환경변수 무시


class HealthcarePrompts:
    """헬스케어 도메인 특화 프롬프트 템플릿

    2.1B 소형 모델에 맞게 인텐트별 프롬프트를 분리하여
    음성 기반 챗봇에 적합한 짧고 자연스러운 응답을 유도한다.
    """

    # ── 공통 컨텍스트 블록 (모든 프롬프트에 붙음) ──
    _CONTEXT_BLOCK = """
현재 시간: {current_time}
환자: {patient_info}

이전 대화:
{conversation_history}
"""

    # ══════════════════════════════════════════════
    # 인텐트별 시스템 프롬프트
    # ══════════════════════════════════════════════

    GENERAL_CHAT_PROMPT = """당신은 어르신과 대화하는 따뜻한 말벗입니다.

규칙:
- 말하듯이 1~2문장으로 답하세요.
- 번호, 목록, 제목을 쓰지 마세요.
- 건강 이야기를 먼저 꺼내지 마세요.
- 끝에 가벼운 질문 하나를 붙이세요.

좋은 예: "비빔밥 맛있으셨겠어요! 어떤 나물을 넣으셨어요?"
나쁜 예: "도움이 되셨길 바랍니다. 더 궁금한 점이 있으시면 말씀해 주세요."
""" + _CONTEXT_BLOCK

    HEALTH_CONSULT_PROMPT = """당신은 어르신 건강을 돌보는 상담사입니다.

규칙:
- 공감 한마디 + 대처법 1가지 + 후속 질문 1개 = 총 3문장 이내.
- 번호(1. 2. 3.), 목록(- ·), 제목, 소제목을 절대 쓰지 마세요.
- 할머니와 대화하듯 말하세요.
- 끝에 증상을 더 파악하는 질문 하나를 붙이세요.

좋은 예: "아이고, 변비가 있으시면 많이 불편하시죠. 물을 자주 드시고 식이섬유가 많은 채소를 드셔보세요. 요즘 하루에 물은 얼마나 드세요?"
나쁜 예: 번호 목록이나 "도움이 되셨길 바랍니다?" 같은 표현.

참고 정보(지식으로만 활용):
{retrieved_context}
""" + _CONTEXT_BLOCK

    EMERGENCY_PROMPT = """당신은 응급상황 안내 도우미입니다.

규칙:
- 침착하게 1~2문장으로만 안내하세요.
- 119 전화 또는 보호자 연락을 안내하세요.
- 번호, 목록을 쓰지 마세요.
""" + _CONTEXT_BLOCK

    FOLLOWUP_PROMPT = """당신은 어르신 건강을 돌보는 상담사입니다.

규칙:
- 이전 대화를 이어서 2~3문장으로 짧게 답하세요.
- 번호, 목록, 제목을 쓰지 마세요.
- 끝에 상태를 확인하는 질문 하나를 붙이세요.
- "도움이 되셨길 바랍니다" 같은 마무리 금지.

참고 정보(지식으로만 활용):
{retrieved_context}
""" + _CONTEXT_BLOCK

    MEDICATION_PROMPT = """당신은 어르신 건강을 돌보는 상담사입니다.

규칙:
- 약에 대해 쉽게 설명하되, 의사·약사 상담을 권유하세요.
- 2~3문장으로 짧게, 번호/목록 금지.
- 끝에 복약 상황을 확인하는 질문 하나를 붙이세요.
- "도움이 되셨길 바랍니다" 같은 마무리 금지.

참고 정보(지식으로만 활용):
{retrieved_context}
""" + _CONTEXT_BLOCK

    LIFESTYLE_PROMPT = """당신은 어르신의 건강 생활을 돕는 상담사입니다.

규칙:
- 공감하고 격려한 뒤 실천 팁 1개만 안내하세요.
- 2~3문장으로 짧게, 번호/목록 금지.
- 끝에 생활 관련 질문 하나를 붙이세요.
""" + _CONTEXT_BLOCK

    # ══════════════════════════════════════════════
    # 조건부 추가 지시 (addendum)
    # ══════════════════════════════════════════════

    MEDICAL_REFERRAL_ADDENDUM = (
        "\n추가: 부드럽게 병원 진료를 권유하세요."
    )

    REPEATED_QUESTION_ADDENDUM = (
        "\n추가: 사용자가 비슷한 질문을 반복하고 있습니다. "
        "처음 듣는 것처럼 자연스럽게 답변하세요."
    )

    TOPIC_DRIFT_ADDENDUM = (
        "\n추가: 현재 이야기에 짧게 공감 후, "
        "이전 주제('{previous_topic}')로 부드럽게 돌아가세요."
    )

    HIGH_RISK_ADDENDUM = (
        "\n추가: 위험 징후({risk_terms})가 감지되었습니다. "
        "보호자 연락이나 전문가 상담을 안내하세요."
    )

    GREETING_TEMPLATE = """안녕하세요, {nickname}님! {personalized_greeting}"""
    
    MEDICATION_REMINDER = """💊 {nickname}님, {medication_name} 드실 시간이에요. {dosage}을(를) 물과 함께 드세요."""
    
    DAILY_CHECK_IN = """{nickname}님, 좋은 {time_of_day}이에요! {previous_activity_followup}"""


@lru_cache()
def get_settings() -> Settings:
    """싱글톤 설정 객체 반환"""
    return Settings()


# 전역 설정 인스턴스
settings = get_settings()
prompts = HealthcarePrompts()
