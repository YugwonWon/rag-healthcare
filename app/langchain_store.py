"""
LangChain 기반 데이터 저장소 모듈
- 벡터 검색 (pgvector)
- 대화 기록 관리 (PostgreSQL)
- 프로필 저장 (PostgreSQL)

LLM은 Kanana(Ollama) 사용, 데이터 레이어만 LangChain 활용
"""

import asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

import asyncpg
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


class LangChainDataStore:
    """
    LangChain 기반 통합 데이터 저장소
    
    기능:
    1. 문서 벡터 검색 (RAG) - pgvector
    2. 대화 기록 저장/조회 - PostgresChatMessageHistory
    3. 프로필 관리 - asyncpg (직접 SQL)
    """
    
    _instance: Optional["LangChainDataStore"] = None
    
    def __new__(cls, *args, **kwargs):
        """싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, connection_string: Optional[str] = None):
        if self._initialized:
            return
            
        # database_url 프로퍼티 사용 (SECRET_MANAGER 지원)
        self.connection_string = connection_string or settings.database_url
        
        if not self.connection_string:
            logger.warning("DATABASE_URL이 설정되지 않음. ChromaDB 폴백 모드로 동작합니다.")
            self._use_postgres = False
            self._initialized = True
            return
        
        self._use_postgres = True
        
        # 1. 임베딩 모델 (로컬 - 변경 없음)
        logger.info(f"임베딩 모델 로딩: {settings.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": settings.EMBEDDING_DEVICE}
        )
        
        # 2. 벡터 스토어 (pgvector)
        self.vectorstore = PGVector(
            connection=self.connection_string,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embeddings=self.embeddings,
        )
        
        # 3. asyncpg 연결 풀 (프로필 등 직접 SQL용)
        self._pool: Optional[asyncpg.Pool] = None
        
        self._initialized = True
        logger.info("LangChain 데이터 스토어 초기화 완료")
    
    @property
    def is_postgres_enabled(self) -> bool:
        return self._use_postgres
    
    # ==========================================
    # 연결 관리
    # ==========================================
    
    def _parse_connection_for_asyncpg(self) -> dict:
        """asyncpg용 연결 파라미터 파싱 (Cloud SQL Unix 소켓 지원)"""
        # Cloud SQL Unix 소켓 형식: postgresql://user:pass@/dbname?host=/cloudsql/project:region:instance
        if "?host=/cloudsql/" in self.connection_string:
            # Unix 소켓 연결
            import re
            match = re.match(
                r'postgresql://([^:]+):([^@]+)@/([^?]+)\?host=(.+)',
                self.connection_string
            )
            if match:
                return {
                    "user": match.group(1),
                    "password": match.group(2),
                    "database": match.group(3),
                    "host": match.group(4),  # /cloudsql/... 경로
                }
        # TCP 연결 (로컬 개발용)
        return {"dsn": self.connection_string}
    
    async def init_pool(self):
        """asyncpg 연결 풀 초기화"""
        if not self._use_postgres:
            return
            
        if self._pool is None:
            conn_params = self._parse_connection_for_asyncpg()
            self._pool = await asyncpg.create_pool(
                **conn_params,
                min_size=1,
                max_size=5
            )
            logger.info("asyncpg 연결 풀 생성됨")
    
    async def close_pool(self):
        """연결 풀 종료"""
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    @asynccontextmanager
    async def get_connection(self):
        """DB 연결 컨텍스트 매니저"""
        if not self._pool:
            await self.init_pool()
        async with self._pool.acquire() as conn:
            yield conn
    
    # ==========================================
    # 문서 벡터 검색 (RAG)
    # ==========================================
    
    def search_documents(self, query: str, k: int = 5) -> list[dict]:
        """
        RAG용 문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            
        Returns:
            [{"content": str, "metadata": dict, "score": float}, ...]
        """
        if not self._use_postgres:
            return []
        
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                }
                for doc, score in docs_with_scores
            ]
        except Exception as e:
            logger.error(f"문서 검색 오류: {e}")
            return []
    
    def add_documents(self, texts: list[str], metadatas: list[dict] = None):
        """문서 추가"""
        if not self._use_postgres:
            return
            
        try:
            self.vectorstore.add_texts(texts, metadatas=metadatas)
            logger.info(f"문서 {len(texts)}개 추가됨")
        except Exception as e:
            logger.error(f"문서 추가 오류: {e}")
    
    def load_documents_from_directory(self, directory: str) -> int:
        """디렉토리에서 문서 일괄 로드"""
        if not self._use_postgres:
            return 0
            
        try:
            loader = DirectoryLoader(
                directory,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            docs = loader.load()
            
            # 청크 분할
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = splitter.split_documents(docs)
            
            # 벡터스토어에 추가
            self.vectorstore.add_documents(splits)
            logger.info(f"디렉토리에서 {len(splits)}개 청크 로드됨")
            return len(splits)
            
        except Exception as e:
            logger.error(f"문서 로드 오류: {e}")
            return 0
    
    # ==========================================
    # 대화 기록 관리
    # ==========================================
    
    def get_chat_history(self, nickname: str) -> PostgresChatMessageHistory:
        """닉네임별 대화 기록 객체 반환"""
        return PostgresChatMessageHistory(
            connection_string=self.connection_string,
            session_id=nickname,
            table_name="chat_history"
        )
    
    def save_conversation(self, nickname: str, user_msg: str, ai_msg: str):
        """대화 저장"""
        if not self._use_postgres:
            return
            
        try:
            history = self.get_chat_history(nickname)
            history.add_user_message(user_msg)
            history.add_ai_message(ai_msg)
            logger.debug(f"대화 저장: {nickname}")
        except Exception as e:
            logger.error(f"대화 저장 오류: {e}")
    
    def get_recent_conversations(self, nickname: str, limit: int = 10) -> list[dict]:
        """최근 대화 조회"""
        if not self._use_postgres:
            return []
            
        try:
            history = self.get_chat_history(nickname)
            messages = history.messages[-(limit*2):]  # user+ai 쌍
            return [
                {"role": "user" if m.type == "human" else "assistant", "content": m.content}
                for m in messages
            ]
        except Exception as e:
            logger.error(f"대화 조회 오류: {e}")
            return []
    
    def clear_conversation_history(self, nickname: str) -> int:
        """대화 기록 삭제"""
        if not self._use_postgres:
            return 0
            
        try:
            history = self.get_chat_history(nickname)
            count = len(history.messages)
            history.clear()
            logger.info(f"대화 기록 삭제: {nickname}, {count}개")
            return count // 2  # 대화 쌍 수
        except Exception as e:
            logger.error(f"대화 삭제 오류: {e}")
            return 0
    
    def get_conversation_count(self, nickname: str) -> int:
        """대화 수 조회 (user+assistant 쌍 기준)"""
        if not self._use_postgres:
            return 0
        try:
            history = self.get_chat_history(nickname)
            return len(history.messages) // 2
        except Exception as e:
            logger.error(f"대화 수 조회 오류: {e}")
            return 0
    
    # ==========================================
    # 대화 요약 관리
    # ==========================================
    
    async def get_conversation_summary(self, nickname: str) -> Optional[str]:
        """저장된 대화 요약 조회"""
        if not self._use_postgres:
            return None
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow(
                    """SELECT summary, summarized_count FROM conversation_summaries 
                       WHERE nickname = $1 ORDER BY updated_at DESC LIMIT 1""",
                    nickname
                )
                if row:
                    return {
                        "summary": row["summary"],
                        "summarized_count": row["summarized_count"]
                    }
                return None
        except Exception as e:
            logger.error(f"요약 조회 오류: {e}")
            return None
    
    async def save_conversation_summary(self, nickname: str, summary: str, summarized_count: int) -> bool:
        """대화 요약 저장"""
        if not self._use_postgres:
            return False
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO conversation_summaries (nickname, summary, summarized_count, updated_at)
                    VALUES ($1, $2, $3, NOW())
                    ON CONFLICT (nickname) DO UPDATE SET
                        summary = EXCLUDED.summary,
                        summarized_count = EXCLUDED.summarized_count,
                        updated_at = NOW()
                """, nickname, summary, summarized_count)
                logger.info(f"대화 요약 저장: {nickname}, {summarized_count}개 대화 요약")
                return True
        except Exception as e:
            logger.error(f"요약 저장 오류: {e}")
            return False
    
    def get_conversations_for_summary(self, nickname: str, start_idx: int = 0, end_idx: int = 10) -> list[dict]:
        """요약할 대화 범위 조회 (오래된 것부터)"""
        if not self._use_postgres:
            return []
        try:
            history = self.get_chat_history(nickname)
            messages = history.messages[start_idx*2:end_idx*2]  # user+ai 쌍
            return [
                {"role": "user" if m.type == "human" else "assistant", "content": m.content}
                for m in messages
            ]
        except Exception as e:
            logger.error(f"요약용 대화 조회 오류: {e}")
            return []
    
    async def should_summarize(self, nickname: str, threshold: int = 10) -> bool:
        """요약이 필요한지 확인 (threshold개 이상의 새 대화가 쌓이면 True)"""
        if not self._use_postgres:
            return False
        
        try:
            current_count = self.get_conversation_count(nickname)
            summary_info = await self.get_conversation_summary(nickname)
            
            if summary_info:
                # 이미 요약된 대화 수 이후로 threshold개 이상 새 대화가 있으면
                new_conversations = current_count - summary_info["summarized_count"]
                return new_conversations >= threshold
            else:
                # 요약이 없으면 threshold개 이상일 때
                return current_count >= threshold
        except Exception as e:
            logger.error(f"요약 필요 여부 확인 오류: {e}")
            return False

    # ==========================================
    # 프로필 관리 (직접 SQL)
    # ==========================================
    
    async def save_profile(self, nickname: str, profile: dict) -> bool:
        """프로필 저장/업데이트 (Upsert)"""
        if not self._use_postgres:
            return False
            
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO profiles (nickname, name, age, conditions, emergency_contact, notes, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, NOW())
                    ON CONFLICT (nickname) DO UPDATE SET
                        name = COALESCE(EXCLUDED.name, profiles.name),
                        age = COALESCE(EXCLUDED.age, profiles.age),
                        conditions = COALESCE(EXCLUDED.conditions, profiles.conditions),
                        emergency_contact = COALESCE(EXCLUDED.emergency_contact, profiles.emergency_contact),
                        notes = COALESCE(EXCLUDED.notes, profiles.notes),
                        updated_at = NOW()
                """, 
                    nickname,
                    profile.get("name"),
                    profile.get("age"),
                    profile.get("conditions"),
                    profile.get("emergency_contact"),
                    profile.get("notes")
                )
                logger.info(f"프로필 저장: {nickname}")
                return True
        except Exception as e:
            logger.error(f"프로필 저장 오류: {e}")
            return False
    
    async def get_profile(self, nickname: str) -> dict:
        """프로필 조회"""
        if not self._use_postgres:
            return {}
            
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM profiles WHERE nickname = $1",
                    nickname
                )
                if row:
                    return dict(row)
                return {}
        except Exception as e:
            logger.error(f"프로필 조회 오류: {e}")
            return {}
    
    async def delete_profile(self, nickname: str) -> bool:
        """프로필 삭제"""
        if not self._use_postgres:
            return False
            
        try:
            async with self.get_connection() as conn:
                await conn.execute(
                    "DELETE FROM profiles WHERE nickname = $1",
                    nickname
                )
                return True
        except Exception as e:
            logger.error(f"프로필 삭제 오류: {e}")
            return False
    
    # ==========================================
    # 통계
    # ==========================================
    
    async def get_stats(self) -> dict:
        """저장소 통계"""
        if not self._use_postgres:
            return {"postgres_enabled": False}
            
        try:
            async with self.get_connection() as conn:
                doc_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM langchain_pg_embedding"
                )
                profile_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM profiles"
                )
                conv_count = await conn.fetchval(
                    "SELECT COUNT(DISTINCT session_id) FROM chat_history"
                )
                
                return {
                    "postgres_enabled": True,
                    "documents": doc_count or 0,
                    "profiles": profile_count or 0,
                    "conversations": conv_count or 0
                }
        except Exception as e:
            logger.error(f"통계 조회 오류: {e}")
            return {"postgres_enabled": True, "error": str(e)}


# ==========================================
# DB 스키마 초기화 (최초 1회)
# ==========================================

INIT_SCHEMA_SQL = """
-- pgvector 확장 (Cloud SQL에서 수동 실행 필요할 수 있음)
CREATE EXTENSION IF NOT EXISTS vector;

-- 프로필 테이블
CREATE TABLE IF NOT EXISTS profiles (
    nickname VARCHAR(100) PRIMARY KEY,
    name VARCHAR(100),
    age INT,
    conditions TEXT,
    emergency_contact VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 프로필 인덱스
CREATE INDEX IF NOT EXISTS idx_profiles_updated ON profiles(updated_at DESC);

-- 대화 요약 테이블
CREATE TABLE IF NOT EXISTS conversation_summaries (
    nickname VARCHAR(100) PRIMARY KEY,
    summary TEXT NOT NULL,
    summarized_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 요약 인덱스
CREATE INDEX IF NOT EXISTS idx_summaries_updated ON conversation_summaries(updated_at DESC);

-- chat_history 테이블은 LangChain이 자동 생성
-- langchain_pg_embedding 테이블도 LangChain이 자동 생성
"""


async def init_database_schema(connection_string: str):
    """데이터베이스 스키마 초기화"""
    try:
        conn = await asyncpg.connect(connection_string)
        await conn.execute(INIT_SCHEMA_SQL)
        await conn.close()
        logger.info("데이터베이스 스키마 초기화 완료")
    except Exception as e:
        logger.error(f"스키마 초기화 오류: {e}")


# ==========================================
# 싱글톤 인스턴스
# ==========================================

_langchain_store: Optional[LangChainDataStore] = None


def get_langchain_store() -> LangChainDataStore:
    """LangChain 데이터 스토어 싱글톤 반환"""
    global _langchain_store
    if _langchain_store is None:
        _langchain_store = LangChainDataStore()
    return _langchain_store
