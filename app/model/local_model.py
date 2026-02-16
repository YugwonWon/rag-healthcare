"""
로컬 LLM 핸들러 (Ollama/Qwen 2.5)
온디바이스 임베딩 모델 포함
"""

import asyncio
from typing import Optional
import httpx
import numpy as np

from app.config import settings
from app.logger import get_logger
from .model_factory import BaseLLM

logger = get_logger(__name__)

# Cloud Run CPU 환경에서 LLM 응답 시간 고려 (최대 5분)
OLLAMA_TIMEOUT = 300.0


class OllamaClient:
    """Ollama API 클라이언트"""
    
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """HTTP 클라이언트 가져오기"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=OLLAMA_TIMEOUT)
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
                "repeat_penalty": 1.3,  # 반복 방지 (파인튜닝 모델 과적합 보완)
                "top_p": 0.9,
                "top_k": 40,
            }
        }
        
        try:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
            
            # EXAONE 4.0 후처리
            content = self._postprocess_exaone(content)
            
            return content
        
        except httpx.TimeoutException:
            logger.error(f"Ollama 타임아웃 ({OLLAMA_TIMEOUT}초 초과)")
            return "죄송합니다, 응답 생성에 시간이 오래 걸리고 있어요. 잠시 후 다시 말씀해 주세요. 🙏"
        
        except httpx.HTTPError as e:
            logger.error(f"Ollama HTTP 에러: {e}")
            return "죄송합니다, 일시적인 오류가 발생했어요. 다시 말씀해 주세요. 🙏"
    
    @staticmethod
    def _postprocess_exaone(content: str) -> str:
        """EXAONE 4.0 응답 후처리: think 블록 제거, 아티팩트 정리"""
        import re
        
        # 1) <think>...</think> 블록 제거 (비추론 모드 빈 블록 포함)
        if "</think>" in content:
            content = content.split("</think>", 1)[1]
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = re.sub(r"</?think>", "", content)
        
        # 2) <tool_call> 등 hallucinated 태그 제거
        content = re.sub(r"</?tool_call[^>]*>", "", content)
        
        # 3) 앞뒤 불필요한 문자 정리 (`:`, `[-1]` 등)
        content = content.strip()
        content = re.sub(r"^\[[-\d]*\]\s*", "", content)  # [-1] 등
        content = re.sub(r"^:\s*", "", content)  # 앞의 : 제거
        
        return content.strip()
    
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
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            # lazy import + safetensors LOAD REPORT 억제 (fd 리다이렉트)
            import os as _os
            _devnull = _os.open(_os.devnull, _os.O_WRONLY)
            _old_stdout = _os.dup(1)
            _old_stderr = _os.dup(2)
            _os.dup2(_devnull, 1)
            _os.dup2(_devnull, 2)
            try:
                from sentence_transformers import SentenceTransformer
                self.__class__._model = SentenceTransformer(
                    settings.EMBEDDING_MODEL,
                    device=settings.EMBEDDING_DEVICE
                )
            finally:
                _os.dup2(_old_stdout, 1)
                _os.dup2(_old_stderr, 2)
                _os.close(_old_stdout)
                _os.close(_old_stderr)
                _os.close(_devnull)
    
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
