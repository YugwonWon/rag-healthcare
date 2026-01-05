"""
ChromaDB 핸들러
치매노인 헬스케어 문서 및 대화 기록 벡터 저장소
"""

import os
from datetime import datetime
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.model.local_model import get_embedding_model, LocalEmbedding
from app.logger import get_logger

logger = get_logger(__name__)


class ChromaHandler:
    """ChromaDB 벡터 저장소 핸들러"""
    
    _instance: Optional["ChromaHandler"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._embedding_model: LocalEmbedding = get_embedding_model()
        
        # ChromaDB 클라이언트 초기화
        if settings.CHROMA_IN_MEMORY:
            self._client = chromadb.Client()
        else:
            persist_dir = settings.CHROMA_PERSIST_DIR
            os.makedirs(persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        
        # 컬렉션 초기화
        self._docs_collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"description": "헬스케어 문서 (치매 관련 정보)"}
        )
        
        self._conversations_collection = self._client.get_or_create_collection(
            name=settings.CONVERSATION_COLLECTION_NAME,
            metadata={"description": "환자별 대화 기록"}
        )
        
        self._patient_profiles_collection = self._client.get_or_create_collection(
            name=settings.PATIENT_PROFILE_COLLECTION,
            metadata={"description": "환자 프로필 정보"}
        )
        
        self._initialized = True
    
    def add_documents(
        self,
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None
    ) -> None:
        """
        헬스케어 문서 추가
        
        Args:
            documents: 문서 텍스트 리스트
            metadatas: 메타데이터 리스트
            ids: 문서 ID 리스트
        """
        if not documents:
            return
        
        embeddings = self._embedding_model.embed_documents(documents)
        
        if ids is None:
            ids = [f"doc_{i}_{datetime.now().timestamp()}" for i in range(len(documents))]
        
        if metadatas is None:
            metadatas = [{"added_at": datetime.now().isoformat()} for _ in documents]
        
        self._docs_collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def search_documents(
        self,
        query: str,
        n_results: int = None,
        where: Optional[dict] = None
    ) -> dict:
        """
        문서 검색
        
        Args:
            query: 검색 쿼리
            n_results: 반환할 결과 수
            where: 필터 조건
        
        Returns:
            검색 결과 딕셔너리
        """
        n_results = n_results or settings.RAG_TOP_K
        query_embedding = self._embedding_model.embed_query(query)
        
        results = self._docs_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def add_conversation(
        self,
        nickname: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[dict] = None
    ) -> str:
        """
        대화 기록 추가 (개인화용)
        
        Args:
            nickname: 사용자 닉네임
            user_message: 사용자 메시지
            assistant_response: AI 응답
            metadata: 추가 메타데이터
        
        Returns:
            대화 기록 ID
        """
        conversation_text = f"사용자: {user_message}\nAI: {assistant_response}"
        embedding = self._embedding_model.embed_query(conversation_text)
        
        timestamp = datetime.now()
        conv_id = f"conv_{nickname}_{timestamp.timestamp()}"
        
        # 메타데이터 정규화 (ChromaDB는 list를 지원하지 않음)
        normalized_metadata = {}
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, list):
                    normalized_metadata[key] = ",".join(str(v) for v in value)
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    normalized_metadata[key] = value
                else:
                    normalized_metadata[key] = str(value)
        
        conv_metadata = {
            "nickname": nickname,
            "timestamp": timestamp.isoformat(),
            "date": timestamp.strftime("%Y-%m-%d"),
            "time": timestamp.strftime("%H:%M:%S"),
            "user_message": user_message[:500],  # 메타데이터 크기 제한
            **normalized_metadata
        }
        
        self._conversations_collection.add(
            documents=[conversation_text],
            embeddings=[embedding],
            metadatas=[conv_metadata],
            ids=[conv_id]
        )
        
        return conv_id
    
    def get_user_conversations(
        self,
        nickname: str,
        n_results: int = None,
        query: Optional[str] = None
    ) -> dict:
        """
        사용자별 대화 기록 조회
        
        Args:
            nickname: 사용자 닉네임
            n_results: 반환할 결과 수
            query: 관련 대화 검색 쿼리 (없으면 최근 대화 반환)
        
        Returns:
            대화 기록 딕셔너리
        """
        n_results = n_results or settings.MAX_CONVERSATION_HISTORY
        
        if query:
            query_embedding = self._embedding_model.embed_query(query)
            results = self._conversations_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"nickname": nickname}
            )
        else:
            # 최근 대화 조회 (시간순 정렬은 별도 처리 필요)
            results = self._conversations_collection.get(
                where={"nickname": nickname},
                limit=n_results
            )
        
        return results
    
    def get_recent_activities(self, nickname: str, hours: int = 24) -> list[dict]:
        """
        최근 활동 내역 조회 (개인화된 인사말 생성용)
        
        Args:
            nickname: 사용자 닉네임
            hours: 조회할 시간 범위
        
        Returns:
            최근 활동 리스트
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        results = self._conversations_collection.get(
            where={
                "nickname": nickname,
            }
        )
        
        activities = []
        if results and results.get("metadatas"):
            for i, metadata in enumerate(results["metadatas"]):
                try:
                    timestamp = datetime.fromisoformat(metadata.get("timestamp", ""))
                    if timestamp >= cutoff_time:
                        activities.append({
                            "timestamp": timestamp,
                            "message": metadata.get("user_message", ""),
                            "document": results["documents"][i] if results.get("documents") else ""
                        })
                except (ValueError, TypeError):
                    continue
        
        # 시간순 정렬
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        return activities
    
    def save_patient_profile(
        self,
        nickname: str,
        profile_data: dict
    ) -> None:
        """
        환자 프로필 저장
        
        Args:
            nickname: 환자 닉네임
            profile_data: 프로필 데이터
        """
        profile_text = f"환자 프로필: {nickname}\n"
        for key, value in profile_data.items():
            profile_text += f"- {key}: {value}\n"
        
        embedding = self._embedding_model.embed_query(profile_text)
        
        # 기존 프로필 삭제 후 추가
        try:
            self._patient_profiles_collection.delete(
                where={"nickname": nickname}
            )
        except Exception:
            pass
        
        self._patient_profiles_collection.add(
            documents=[profile_text],
            embeddings=[embedding],
            metadatas=[{"nickname": nickname, **profile_data}],
            ids=[f"profile_{nickname}"]
        )
    
    def get_patient_profile(self, nickname: str) -> Optional[dict]:
        """
        환자 프로필 조회
        
        Args:
            nickname: 환자 닉네임
        
        Returns:
            프로필 데이터 또는 None
        """
        results = self._patient_profiles_collection.get(
            where={"nickname": nickname},
            limit=1
        )
        
        if results and results.get("metadatas"):
            return results["metadatas"][0]
        return None
    
    def get_collection_stats(self) -> dict:
        """컬렉션 통계 조회"""
        return {
            "documents": self._docs_collection.count(),
            "conversations": self._conversations_collection.count(),
            "patient_profiles": self._patient_profiles_collection.count()
        }


def get_chroma_handler() -> ChromaHandler:
    """ChromaDB 핸들러 인스턴스 가져오기"""
    return ChromaHandler()
