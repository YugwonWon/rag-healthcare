"""
OpenAI 모델 핸들러 (Fallback용)
"""

from typing import Optional
from openai import AsyncOpenAI

from app.config import settings
from .model_factory import BaseLLM


class OpenAIModel(BaseLLM):
    """OpenAI GPT 모델 (Fallback용)"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MODEL
        self._client: Optional[AsyncOpenAI] = None
        
        if self.api_key:
            self._client = AsyncOpenAI(api_key=self.api_key)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
        
        Returns:
            생성된 텍스트
        """
        if not self._client:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)
    
    async def chat(self, messages: list[dict], **kwargs) -> str:
        """
        채팅 형식 생성
        
        Args:
            messages: 채팅 메시지 리스트
        
        Returns:
            생성된 응답
        """
        if not self._client:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        
        temperature = kwargs.get("temperature", settings.LLM_TEMPERATURE)
        max_tokens = kwargs.get("max_tokens", settings.LLM_MAX_TOKENS)
        
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    async def is_available(self) -> bool:
        """모델 사용 가능 여부"""
        return self._client is not None and self.api_key is not None
