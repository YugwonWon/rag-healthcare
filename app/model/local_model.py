"""
로컬 LLM 핸들러 (Ollama/Qwen 2.5)
온디바이스 임베딩 모델 포함
"""

import asyncio
from typing import Optional
import httpx
from sentence_transformers import SentenceTransformer
import numpy as np

from app.config import settings
from app.logger import get_logger
from .model_factory import BaseLLM

logger = get_logger(__name__)


class OllamaClient:
    """Ollama API 클라이언트"""
    
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 가져오기"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client
    
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
        
        Returns:
            생성된 텍스트
        """
        client = await self._get_client()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or settings.LLM_TEMPERATURE,
                "num_predict": max_tokens or settings.LLM_MAX_TOKENS,
            }
        }
        
        response = await client.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        response.raise_for_status()
        
        return response.json()["response"]
    
    async def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        채팅 형식 생성
        
        Args:
            messages: 채팅 메시지 리스트
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
        
        Returns:
            생성된 응답
        """
        client = await self._get_client()
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or settings.LLM_TEMPERATURE,
                "num_predict": max_tokens or settings.LLM_MAX_TOKENS,
            }
        }
        
        response = await client.post(
            f"{self.base_url}/api/chat",
            json=payload
        )
        response.raise_for_status()
        
        return response.json()["message"]["content"]
    
    async def is_available(self) -> bool:
        """Ollama 서버 사용 가능 여부 확인"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self):
        """클라이언트 종료"""
        if self._client:
            await self._client.aclose()
            self._client = None


class LocalEmbedding:
    """온디바이스 임베딩 모델 (sentence-transformers)"""
    
    _instance: Optional["LocalEmbedding"] = None
    _model: Optional[SentenceTransformer] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE
            )
    
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        텍스트 임베딩 생성
        
        Args:
            texts: 임베딩할 텍스트 리스트
        
        Returns:
            임베딩 벡터 배열 (shape: [n, 384])
        """
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings
    
    def embed_query(self, query: str) -> list[float]:
        """
        쿼리 임베딩 생성
        
        Args:
            query: 쿼리 텍스트
        
        Returns:
            임베딩 벡터 리스트
        """
        embedding = self._model.encode(query, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        문서 임베딩 생성
        
        Args:
            documents: 문서 리스트
        
        Returns:
            임베딩 벡터 리스트
        """
        embeddings = self._model.encode(documents, convert_to_numpy=True)
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        """임베딩 차원"""
        return settings.EMBEDDING_DIMENSION


class LocalLLM(BaseLLM):
    """로컬 LLM (Ollama 기반)"""
    
    def __init__(self):
        self.client = OllamaClient()
        self._embedding = LocalEmbedding()
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        return await self.client.generate(prompt, **kwargs)
    
    async def chat(self, messages: list[dict], **kwargs) -> str:
        """채팅 형식 생성"""
        return await self.client.chat(messages, **kwargs)
    
    async def is_available(self) -> bool:
        """모델 사용 가능 여부"""
        return await self.client.is_available()
    
    @property
    def embedding(self) -> LocalEmbedding:
        """임베딩 모델"""
        return self._embedding


# 싱글톤 임베딩 인스턴스
def get_embedding_model() -> LocalEmbedding:
    """임베딩 모델 인스턴스 가져오기"""
    return LocalEmbedding()
