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
import warnings
warnings.filterwarnings("ignore", message=".*deprecated.*", module="langchain")
warnings.filterwarnings("ignore", message=".*LangChain.*")

# 임포트 시 safetensors/sentence_transformers가 stdout에 출력하는 것을 억제
import os as _os
_devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
_saved_stdout = _os.dup(1)
_saved_stderr = _os.dup(2)
_os.dup2(_devnull_fd, 1)
_os.dup2(_devnull_fd, 2)
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
finally:
    _os.dup2(_saved_stdout, 1)
    _os.dup2(_saved_stderr, 2)
    _os.close(_saved_stdout)
    _os.close(_saved_stderr)
    _os.close(_devnull_fd)

from langchain_postgres import PGVector
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.logger import get_logger
from app.utils.timezone import get_kst_now, KST

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
            logger.warning("DATABASE_URL이 설정되지 않음. pgvector를 사용할 수 없습니다.")
            self._use_postgres = False
            self._initialized = True
            return
        
        self._use_postgres = True
        
        # 1. 임베딩 모델 (로컬 - 변경 없음)
        logger.info(f"임베딩 모델 로딩: {settings.EMBEDDING_MODEL}")
        # safetensors LOAD REPORT (Rust stdout) + HF Hub 경고 (stderr) 출력 억제
        # os.dup2로 파일 디스크립터 레벨에서 리다이렉트 (Rust/C 출력 포함)
        import os as _os
        _devnull = _os.open(_os.devnull, _os.O_WRONLY)
        _old_stdout = _os.dup(1)
        _old_stderr = _os.dup(2)
        _os.dup2(_devnull, 1)
        _os.dup2(_devnull, 2)
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={"device": settings.EMBEDDING_DEVICE}
            )
        finally:
            _os.dup2(_old_stdout, 1)
            _os.dup2(_old_stderr, 2)
            _os.close(_old_stdout)
            _os.close(_old_stderr)
            _os.close(_devnull)
        
        # 2. 벡터 스토어 (pgvector)
        self.vectorstore = PGVector(
            connection=self.connection_string,
            collection_name="healthcare_rag",
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

    async def ensure_analytics_schema(self) -> bool:
        """연구·분석용 conversation_logs 테이블을 보장한다 (idempotent).

        기존 INIT_SCHEMA_SQL은 CREATE EXTENSION 등 권한이 필요한 구문을 포함해
        한 트랜잭션에서 실패하면 전체가 롤백될 수 있다. 분석 테이블은 그와 무관하게
        항상 생성되도록 startup에서 이 메서드를 따로 호출한다."""
        if not self._use_postgres:
            return False
        try:
            async with self.get_connection() as conn:
                await conn.execute(CONVERSATION_LOGS_SCHEMA_SQL)
            logger.info("conversation_logs 분석 테이블 준비 완료")
            return True
        except Exception as e:
            logger.error(f"분석 스키마 초기화 오류: {e}")
            return False
    
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
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
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

    async def save_conversation_log(
        self,
        nickname: str,
        user_msg: str,
        ai_msg: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """연구·분석용 턴 단위 대화 로그 저장.

        chat_history(LangChain)는 런타임 대화 맥락 복원에 쓰이고 타임스탬프·메타데이터가
        없어 사후 분석이 어렵다. 이 테이블은 KST 타임스탬프와 인텐트/위험도/증상 같은
        파이프라인 산출 메타데이터를 턴마다 함께 남겨 이용자 그룹 분석을 가능하게 한다.
        chat_history와 독립적이므로 한쪽이 실패해도 다른 쪽 저장에는 영향을 주지 않는다.
        """
        if not self._use_postgres:
            return False

        meta = metadata or {}
        import json as _json
        try:
            async with self.get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO conversation_logs (
                        nickname, user_message, ai_message,
                        intent, intent_confidence, risk_level,
                        detected_symptoms, risk_categories,
                        repeated_question, topic_drifted,
                        created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    nickname,
                    user_msg,
                    ai_msg,
                    meta.get("intent"),
                    meta.get("intent_confidence"),
                    meta.get("risk_level"),
                    _json.dumps(meta.get("detected_symptoms", []), ensure_ascii=False),
                    _json.dumps(meta.get("risk_categories", []), ensure_ascii=False),
                    bool(meta.get("repeated_question", False)),
                    bool(meta.get("topic_drifted", False)),
                    get_kst_now(),
                )
            logger.debug(f"대화 로그 저장: {nickname} | intent={meta.get('intent')}")
            return True
        except Exception as e:
            logger.error(f"대화 로그 저장 오류: {e}")
            return False

    # ==========================================
    # 대화 기록 추출 (관리자 CSV 다운로드 / 분석용)
    # ==========================================

    # CSV 컬럼 순서 (엔드포인트·스크립트가 공유)
    CHAT_HISTORY_FIELDS = ["id", "nickname", "role", "content"]
    CONVERSATION_LOG_FIELDS = [
        "id", "nickname", "created_at_kst", "intent", "intent_confidence",
        "risk_level", "detected_symptoms", "risk_categories",
        "repeated_question", "topic_drifted", "user_message", "ai_message",
    ]

    async def _table_exists(self, conn, table: str) -> bool:
        return bool(await conn.fetchval("SELECT to_regclass($1)", table))

    async def fetch_chat_history_rows(self, nickname: Optional[str] = None) -> list[dict]:
        """레거시 chat_history(LangChain JSONB)를 행 단위로 복원한다.

        타임스탬프·메타데이터는 없지만 모든 런타임 대화를 포함한다."""
        if not self._use_postgres:
            return []
        import json as _json
        rows_out: list[dict] = []
        try:
            async with self.get_connection() as conn:
                if not await self._table_exists(conn, "chat_history"):
                    return []
                where = "WHERE session_id = $1" if nickname else ""
                args = [nickname] if nickname else []
                rows = await conn.fetch(
                    f"SELECT id, session_id, message FROM chat_history {where} ORDER BY id ASC",
                    *args,
                )
            for r in rows:
                msg = r["message"]
                if isinstance(msg, str):
                    try:
                        msg = _json.loads(msg)
                    except _json.JSONDecodeError:
                        msg = {}
                mtype = (msg or {}).get("type", "")
                content = ((msg or {}).get("data") or {}).get("content", "")
                rows_out.append({
                    "id": r["id"],
                    "nickname": r["session_id"],
                    "role": "user" if mtype == "human" else ("assistant" if mtype == "ai" else mtype),
                    "content": (content or "").replace("\r", " ").strip(),
                })
        except Exception as e:
            logger.error(f"chat_history 추출 오류: {e}")
        return rows_out

    async def fetch_conversation_log_rows(self, nickname: Optional[str] = None) -> list[dict]:
        """conversation_logs(타임스탬프 + 메타데이터)를 행 단위로 반환한다."""
        if not self._use_postgres:
            return []
        rows_out: list[dict] = []
        try:
            async with self.get_connection() as conn:
                if not await self._table_exists(conn, "conversation_logs"):
                    return []
                where = "WHERE nickname = $1" if nickname else ""
                args = [nickname] if nickname else []
                rows = await conn.fetch(
                    f"""
                    SELECT id, nickname, user_message, ai_message,
                           intent, intent_confidence, risk_level,
                           detected_symptoms, risk_categories,
                           repeated_question, topic_drifted, created_at
                    FROM conversation_logs
                    {where}
                    ORDER BY created_at ASC, id ASC
                    """,
                    *args,
                )
            for r in rows:
                created = r["created_at"]
                created_kst = (
                    created.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S") if created else ""
                )
                rows_out.append({
                    "id": r["id"],
                    "nickname": r["nickname"],
                    "created_at_kst": created_kst,
                    "intent": r["intent"] or "",
                    "intent_confidence": r["intent_confidence"] if r["intent_confidence"] is not None else "",
                    "risk_level": r["risk_level"] or "",
                    "detected_symptoms": r["detected_symptoms"] or "[]",
                    "risk_categories": r["risk_categories"] or "[]",
                    "repeated_question": r["repeated_question"],
                    "topic_drifted": r["topic_drifted"],
                    "user_message": (r["user_message"] or "").replace("\r", " ").strip(),
                    "ai_message": (r["ai_message"] or "").replace("\r", " ").strip(),
                })
        except Exception as e:
            logger.error(f"conversation_logs 추출 오류: {e}")
        return rows_out

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

# 연구·분석용 대화 로그 스키마 (독립 실행 가능 — CREATE EXTENSION 등 권한 이슈와 분리)
# chat_history(LangChain 자동생성)는 타임스탬프·메타데이터가 없어 사후 분석이 어려우므로
# 턴 단위로 KST 타임스탬프 + 파이프라인 산출 메타데이터를 함께 남긴다.
CONVERSATION_LOGS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversation_logs (
    id BIGSERIAL PRIMARY KEY,
    nickname VARCHAR(100) NOT NULL,
    user_message TEXT,
    ai_message TEXT,
    intent VARCHAR(50),
    intent_confidence REAL,
    risk_level VARCHAR(20),
    detected_symptoms JSONB,
    risk_categories JSONB,
    repeated_question BOOLEAN DEFAULT FALSE,
    topic_drifted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 분석 쿼리용 인덱스 (이용자별 / 시간순 / 인텐트별 집계)
CREATE INDEX IF NOT EXISTS idx_conv_logs_nickname ON conversation_logs(nickname);
CREATE INDEX IF NOT EXISTS idx_conv_logs_created ON conversation_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_conv_logs_intent ON conversation_logs(intent);
"""


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
""" + CONVERSATION_LOGS_SCHEMA_SQL + """
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
