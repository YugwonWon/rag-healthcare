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
    
    # LLM 설정 (Ollama/Qwen)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048
    
    # OpenAI 설정 (Fallback용)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    
    # 임베딩 설정 (온디바이스)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_DEVICE: str = "cpu"  # cpu, cuda, mps
    
    # ChromaDB 설정
    CHROMA_PERSIST_DIR: str = "./data/chroma"
    CHROMA_COLLECTION_NAME: str = "healthcare_docs"
    CHROMA_IN_MEMORY: bool = False  # Cloud Run에서는 True
    
    # 대화 기록 설정
    CONVERSATION_COLLECTION_NAME: str = "conversations"
    MAX_CONVERSATION_HISTORY: int = 10
    
    # RAG 설정
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.5
    
    # 헬스케어 도메인 설정
    PATIENT_PROFILE_COLLECTION: str = "patient_profiles"
    MEDICATION_REMINDER_ENABLED: bool = True
    DAILY_ROUTINE_TRACKING: bool = True
    
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
    
    SYSTEM_PROMPT = """당신은 치매노인을 돌보는 따뜻하고 친절한 AI 도우미입니다. 
다음 지침을 따라 대화해주세요:

1. 항상 존댓말을 사용하고, 천천히 명확하게 설명합니다.
2. 복잡한 내용은 짧고 간단한 문장으로 나눠서 전달합니다.
3. 환자의 감정을 존중하고 공감하며 대화합니다.
4. 이전 대화 내용을 자연스럽게 언급하여 연속성을 유지합니다.
5. 복약 시간, 식사, 산책 등 일상 루틴을 부드럽게 상기시킵니다.
6. 위험한 상황이나 건강 이상 징후가 감지되면 보호자/의료진 연락을 권합니다.

환자 정보:
{patient_info}

이전 대화 내용:
{conversation_history}

관련 의료 정보:
{retrieved_context}
"""
    
    GREETING_TEMPLATE = """안녕하세요, {nickname}님! 오늘도 좋은 하루 되고 계신가요?
{personalized_greeting}
무엇을 도와드릴까요?"""
    
    MEDICATION_REMINDER = """💊 {nickname}님, {medication_name} 드실 시간이에요.
{dosage}을(를) 물과 함께 드시면 됩니다.
복용하셨으면 '먹었어요'라고 말씀해 주세요."""
    
    DAILY_CHECK_IN = """🌤️ {nickname}님, 좋은 {time_of_day}이에요!
{previous_activity_followup}
오늘 기분은 어떠세요?"""


@lru_cache()
def get_settings() -> Settings:
    """싱글톤 설정 객체 반환"""
    return Settings()


# 전역 설정 인스턴스
settings = get_settings()
prompts = HealthcarePrompts()
