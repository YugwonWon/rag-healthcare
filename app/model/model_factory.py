"""
LLM 팩토리 패턴 구현
"""

from typing import Optional, Protocol
from abc import ABC, abstractmethod

from app.config import settings


class LLMInterface(Protocol):
    """LLM 인터페이스 프로토콜"""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        ...
    
    async def chat(self, messages: list[dict], **kwargs) -> str:
        """채팅 형식 생성"""
        ...


class BaseLLM(ABC):
    """LLM 기본 추상 클래스"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        pass
    
    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> str:
        """채팅 형식 생성"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """모델 사용 가능 여부"""
        pass


class ModelFactory:
    """LLM 모델 팩토리"""
    
    _instance: Optional[BaseLLM] = None
    
    @classmethod
    def get_model(cls, force_type: Optional[str] = None) -> BaseLLM:
        """
        LLM 모델 인스턴스 반환
        
        Args:
            force_type: 강제할 모델 타입 ('local', 'openai')
        
        Returns:
            LLM 모델 인스턴스
        """
        if cls._instance is not None and force_type is None:
            return cls._instance
        
        if force_type == "openai" or (force_type is None and settings.OPENAI_API_KEY):
            from .openai_model import OpenAIModel
            try:
                model = OpenAIModel()
                if model.is_available():
                    cls._instance = model
                    return model
            except Exception:
                pass
        
        # 기본값: 로컬 모델 (Ollama)
        from .local_model import LocalLLM
        model = LocalLLM()
        cls._instance = model
        return model
    
    @classmethod
    def reset(cls):
        """팩토리 인스턴스 초기화"""
        cls._instance = None


def get_llm() -> BaseLLM:
    """LLM 인스턴스 가져오기 (편의 함수)"""
    return ModelFactory.get_model()
