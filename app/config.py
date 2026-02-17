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
    # - exaone-counseling: LGAI-EXAONE/EXAONE-4.0-1.2B (현재 사용)
    #   └ 특징: 1.28B 경량, 온디바이스 최적화, 한국어 지원
    #   └ GGUF: models/EXAONE-4.0-1.2B-Q4_K_M.gguf
    #   └ CPU 추론 속도 우수 (1.2B 경량)
    #
    # - kanana-counseling: Kakao Kanana-nano 2.1B Instruct (이전 모델)
    #   └ HuggingFace: kakaocorp/kanana-nano-2.1b-instruct
    #   └ GGUF: kanana-nano-2.1b-instruct-q4_k_m.gguf
    #
    # - qwen2.5:3b: Alibaba Qwen 2.5 3B
    #   └ Ollama 공식 모델
    # ============================================================
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "exaone-counseling"  # EXAONE 4.0 1.2B 파인튜닝된 상담 모델
    LLM_TEMPERATURE: float = 0.1  # EXAONE 비추론 모드: 한국어 code-switching 방지에 0.1 권장
    LLM_MAX_TOKENS: int = 256  # 짧은 응답을 위해 1024 -> 256
    
    # OpenAI 설정 (Fallback용)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    
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
    """헬스케어 도메인 특화 프롬프트 템플릿"""
    
    SYSTEM_PROMPT = """당신은 치매노인을 돌보는 따뜻한 AI 도우미입니다.

## 응답 원칙 (중요!)
- **3~4문장으로 짧게** 답변하세요.
- 핵심 정보만 전달합니다.
- 마지막에 궁금한 점이나 필요한 것을 추가로 더 묻지 않기.

## 대화 스타일
- 존댓말 사용, 간결하고 명확하게
- 환자의 말에 공감하며 자연스럽게 대화
- 일상 대화에는 자연스럽게 일상 대화로 답변
- 위험 징후 감지 시에만 보호자 연락 권유

## 현재 시간: {current_time}

## 환자 정보
{patient_info}

## 이전 대화
{conversation_history}

## 참고 정보(직접 인용하지 말고 지식으로만 활용)
{retrieved_context}
"""
    
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
