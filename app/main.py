"""
FastAPI 메인 서버
치매노인 맞춤형 헬스케어 RAG 챗봇 API
"""

import time
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.utils import get_kst_now, get_kst_datetime_str
from app.model import get_llm
from app.langchain_store import get_langchain_store
from app.retriever import get_query_handler
from app.healthcare import SymptomTracker, MedicationReminder, DailyRoutineManager
from app.logger import init_logging, get_logger, log_startup_info, log_request, log_response

# 로깅 초기화
init_logging()
logger = get_logger(__name__)

# LangChain 스토어 (pgvector)
_langchain_store = None

def get_store():
    global _langchain_store
    if _langchain_store is None:
        _langchain_store = get_langchain_store()
    return _langchain_store


# 요청/응답 모델
class ChatRequest(BaseModel):
    """채팅 요청"""
    nickname: str = Field(..., description="사용자 닉네임", min_length=1, max_length=50)
    message: str = Field(..., description="사용자 메시지", min_length=1)
    include_history: bool = Field(default=True, description="대화 기록 포함 여부")


class ChatResponse(BaseModel):
    """채팅 응답"""
    response: str
    nickname: str
    timestamp: str
    intent: Optional[str] = None  # LangGraph 의도 분류 결과
    symptom_alert: Optional[dict] = None
    medication_reminders: Optional[list[str]] = None
    routine_status: Optional[str] = None
    health_analysis: Optional[dict] = None  # NER + N-gram 기반 건강 분석 결과
    emergency_alert: Optional[dict] = None  # 위급 상황 알림


class GreetingRequest(BaseModel):
    """인사말 요청"""
    nickname: str = Field(..., description="사용자 닉네임")


class GreetingResponse(BaseModel):
    """인사말 응답"""
    greeting: str
    nickname: str
    timestamp: str
    suggestions: list[str] = []


class PatientProfileRequest(BaseModel):
    """환자 프로필 요청"""
    nickname: str
    name: Optional[str] = None
    age: Optional[int] = None
    conditions: Optional[str] = None  # 쉼표로 구분된 상태/질환
    emergency_contact: Optional[str] = None
    notes: Optional[str] = None


class DocumentRequest(BaseModel):
    """문서 추가 요청"""
    documents: list[str]
    metadatas: Optional[list[dict]] = None


class HealthStatusResponse(BaseModel):
    """건강 상태 응답"""
    status: str
    stats: dict
    llm_available: bool


# 전역 인스턴스
symptom_tracker = SymptomTracker()
medication_reminder = MedicationReminder()
routine_manager = DailyRoutineManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시 로그 기록
    config_info = {
        "OLLAMA_MODEL": settings.OLLAMA_MODEL,
        "EMBEDDING_MODEL": settings.EMBEDDING_MODEL,
        "RAG_TOP_K": settings.RAG_TOP_K,
    }
    log_startup_info(logger, settings.APP_NAME, settings.APP_VERSION, config_info)
    
    # LangChain store 연결 풀 초기화 (PostgreSQL)
    store = get_store()
    await store.init_pool()
    logger.info("🐘 PostgreSQL 연결 풀 초기화 완료")
    
    # pgvector에 문서가 0개이면 healthcare_docs 자동 로드
    try:
        pg_stats = await store.get_stats()
        doc_count = pg_stats.get("documents", 0)
        logger.info(f"📚 pgvector 문서 수: {doc_count}")
        
        if doc_count == 0:
            from pathlib import Path
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.document_loaders import TextLoader
            
            docs_dir = Path(__file__).parent.parent / "data" / "healthcare_docs"
            conv_dir = Path(__file__).parent.parent / "data" / "conversations"
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200,
                separators=["\n---\n", "\n\n", "\n", " "]
            )
            
            all_docs = []
            
            # healthcare_docs 로드
            if docs_dir.exists():
                txt_files = sorted(docs_dir.glob("*.txt"))
                logger.info(f"📂 healthcare_docs 자동 로드: {len(txt_files)}개 파일")
                for txt_file in txt_files:
                    try:
                        loader = TextLoader(str(txt_file), encoding="utf-8")
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata["category"] = "healthcare_docs"
                            doc.metadata["source_name"] = txt_file.stem
                        all_docs.extend(splitter.split_documents(docs))
                    except Exception as e:
                        logger.warning(f"파일 로드 실패: {txt_file.name} - {e}")
            
            # conversations 로드
            if conv_dir.exists():
                conv_files = sorted(conv_dir.glob("*.txt"))
                logger.info(f"📂 conversations 자동 로드: {len(conv_files)}개 파일")
                for txt_file in conv_files:
                    try:
                        loader = TextLoader(str(txt_file), encoding="utf-8")
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata["category"] = "conversations"
                            doc.metadata["source_name"] = txt_file.stem
                        all_docs.extend(splitter.split_documents(docs))
                    except Exception as e:
                        logger.warning(f"파일 로드 실패: {txt_file.name} - {e}")
            
            # pgvector에 배치 로드
            if all_docs:
                batch_size = 50
                loaded = 0
                for i in range(0, len(all_docs), batch_size):
                    batch = all_docs[i:i + batch_size]
                    try:
                        store.vectorstore.add_documents(batch)
                        loaded += len(batch)
                    except Exception as e:
                        logger.warning(f"배치 로드 실패: {e}")
                logger.info(f"✅ pgvector 자동 로드 완료: {loaded}/{len(all_docs)}개 청크")
    except Exception as e:
        logger.warning(f"pgvector 문서 자동 로드 실패 (비필수): {e}")

    try:
        from app.knowledge_graph.health_kg import get_health_kg
        kg = get_health_kg()
        logger.info(f"🧠 Knowledge Graph 초기화 완료 | 노드={kg.graph.number_of_nodes()}, 엣지={kg.graph.number_of_edges()}")
    except Exception as e:
        logger.warning(f"Knowledge Graph 초기화 실패 (비필수): {e}")
    
    yield
    
    # 종료 시
    store = get_store()
    await store.close_pool()
    logger.info("🐘 PostgreSQL 연결 풀 종료")
    logger.info("👋 서버 종료...")


# FastAPI 앱 초기화
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="치매노인을 위한 맞춤형 헬스케어 RAG 챗봇 API",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 요청/응답 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """모든 HTTP 요청/응답 로깅"""
    start_time = time.time()
    
    # 요청 로깅
    log_request(logger, request.method, request.url.path)
    
    # 요청 처리
    response = await call_next(request)
    
    # 응답 로깅
    duration_ms = (time.time() - start_time) * 1000
    log_response(logger, request.method, request.url.path, response.status_code, duration_ms)
    
    return response


# 의존성
def get_handler():
    return get_query_handler()


# 엔드포인트
@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": f"{settings.APP_NAME}에 오신 것을 환영합니다!",
        "version": settings.APP_VERSION,
        "status": "running"
    }


@app.get("/health", response_model=HealthStatusResponse)
async def health_check():
    """
    헬스 체크 엔드포인트
    """
    stats = {"documents": 0, "conversations": 0, "patient_profiles": 0}
    
    try:
        store = get_store()
        pg_stats = await store.get_stats()
        if pg_stats.get("postgres_enabled") and "error" not in pg_stats:
            stats["documents"] = pg_stats.get("documents", 0)
            stats["conversations"] = pg_stats.get("conversations", 0)
            stats["patient_profiles"] = pg_stats.get("profiles", 0)
            stats["store"] = "pgvector"
    except Exception as e:
        logger.warning(f"pgvector 통계 조회 실패: {e}")
    
    # LLM 가용성 체크
    llm = get_llm()
    llm_available = await llm.is_available()
    
    return HealthStatusResponse(
        status="healthy",
        stats=stats,
        llm_available=llm_available
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    handler=Depends(get_handler)
):
    """
    채팅 엔드포인트
    닉네임 기반 개인화된 대화 처리
    
    NER + N-gram 기반 건강 위험 신호 감지 전처리 적용:
    1. 건강 관련 용어 개체명 인식 (NER)
    2. 태깅된 용어 전후 5단어 N-gram 추출
    3. 규칙 기반 건강 위험 신호 감지
    4. 향상된 쿼리로 RAG 검색 수행
    """
    try:
        # 1. 증상 분석
        symptom_analysis = symptom_tracker.analyze_message(
            request.nickname,
            request.message
        )
        
        # 2. RAG 기반 응답 생성 (NER + N-gram 전처리 포함)
        result = await handler.process_query(
            nickname=request.nickname,
            query=request.message,
            include_history=request.include_history
        )
        
        # 응답 추출 (dict 형태로 반환됨)
        response = result.get("response", "") if isinstance(result, dict) else result
        health_analysis = result.get("health_analysis") if isinstance(result, dict) else None
        intent = result.get("intent") if isinstance(result, dict) else None
        emergency_alert = result.get("emergency_alert") if isinstance(result, dict) else None
        
        # 3. 복약 알림 확인
        med_reminders = medication_reminder.check_and_send_reminders(request.nickname)
        
        # 4. 루틴 상태 (설정에서 활성화된 경우에만)
        routine_status = None
        if settings.DAILY_ROUTINE_TRACKING:
            routine_status = routine_manager.generate_routine_message(request.nickname)
        
        # 5. 위험 증상 감지 시 경고 추가
        if symptom_analysis.get("needs_attention"):
            recommendations = symptom_analysis.get("recommendations", [])
            if recommendations:
                response += "\n\n" + "\n".join(recommendations)
        
        return ChatResponse(
            response=response,
            nickname=request.nickname,
            timestamp=get_kst_now().isoformat(),
            intent=intent,
            symptom_alert=symptom_analysis if symptom_analysis.get("detected_symptoms") else None,
            medication_reminders=med_reminders if med_reminders else None,
            routine_status=routine_status,
            health_analysis=health_analysis,
            emergency_alert=emergency_alert
        )
    
    except Exception as e:
        logger.error(f"채팅 처리 중 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"채팅 처리 중 오류: {str(e)}")


@app.post("/greeting", response_model=GreetingResponse)
async def get_greeting(
    request: GreetingRequest,
    handler=Depends(get_handler)
):
    """
    개인화된 인사말 생성
    이전 대화 기반 후속 질문 포함
    """
    try:
        greeting = await handler.generate_greeting(request.nickname)
        
        # 루틴 추적 활성화된 경우에만 제안 생성
        suggestions = []
        if settings.DAILY_ROUTINE_TRACKING:
            suggestions = routine_manager.get_activity_suggestions(request.nickname)
        
        return GreetingResponse(
            greeting=greeting,
            nickname=request.nickname,
            timestamp=get_kst_now().isoformat(),
            suggestions=suggestions
        )
    
    except Exception as e:
        logger.error(f"인사말 생성 중 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"인사말 생성 중 오류: {str(e)}")


@app.post("/profile")
async def save_profile(request: PatientProfileRequest):
    """
    환자 프로필 저장 (PostgreSQL)
    """
    try:
        profile_data = {
            "nickname": request.nickname,
            "name": request.name,
            "age": request.age,
            "conditions": request.conditions,
            "emergency_contact": request.emergency_contact,
            "notes": request.notes,
            "updated_at": get_kst_now().isoformat()
        }
        
        # None 값 제거
        profile_data = {k: v for k, v in profile_data.items() if v is not None}
        
        # 스토어 저장 (pgvector)
        store = get_store()
        success = await store.save_profile(request.nickname, profile_data)
        if not success:
            raise Exception("PostgreSQL 저장 실패")
        logger.info(f"프로필 저장 (PostgreSQL): {request.nickname}")
        
        # 루틴 초기화
        routine_manager.initialize_routine(request.nickname)
        
        return {
            "status": "success",
            "message": f"{request.nickname}님의 프로필이 저장되었습니다.",
            "profile": profile_data,
            "store": "postgresql"
        }
    
    except Exception as e:
        logger.error(f"프로필 저장 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"프로필 저장 중 오류: {str(e)}")


@app.get("/profile/{nickname}")
async def get_profile(nickname: str):
    """
    환자 프로필 조회 (PostgreSQL)
    """
    store = get_store()
    profile = await store.get_profile(nickname)
    
    if not profile:
        raise HTTPException(status_code=404, detail=f"{nickname}님의 프로필을 찾을 수 없습니다.")
    
    return {"profile": profile, "store": "postgresql"}


@app.get("/history/{nickname}")
async def get_conversation_history(
    nickname: str,
    limit: int = 10,
):
    """
    대화 기록 조회
    """
    store = get_store()
    results = store.get_recent_conversations(nickname, limit=limit)
    
    return {
        "nickname": nickname,
        "conversations": results,
    }


@app.delete("/history/{nickname}")
async def delete_conversation_history(nickname: str):
    """
    사용자의 대화 기록 삭제
    """
    try:
        store = get_store()
        deleted_count = await store.delete_conversations(nickname)
        logger.info(f"대화 기록 삭제 | nickname={nickname} | count={deleted_count}")
        return {
            "success": True,
            "nickname": nickname,
            "deleted_count": deleted_count,
            "message": f"{nickname}님의 대화 기록 {deleted_count}개가 삭제되었습니다."
        }
    except Exception as e:
        logger.error(f"대화 기록 삭제 실패 | nickname={nickname} | error={e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents")
async def add_documents(request: DocumentRequest):
    """
    헬스케어 문서 추가
    """
    try:
        store = get_store()
        from langchain_core.documents import Document
        docs = [
            Document(
                page_content=doc,
                metadata=request.metadatas[i] if request.metadatas and i < len(request.metadatas) else {}
            )
            for i, doc in enumerate(request.documents)
        ]
        store.vectorstore.add_documents(docs)
        
        return {
            "status": "success",
            "message": f"{len(request.documents)}개 문서가 추가되었습니다."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문서 추가 중 오류: {str(e)}")


@app.get("/stats")
async def get_stats():
    """
    시스템 통계 조회
    """
    store = get_store()
    try:
        stats = await store.get_stats()
    except Exception:
        stats = {}
    
    return {
        "database_stats": stats,
        "settings": {
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "llm_model": settings.OLLAMA_MODEL,
            "rag_top_k": settings.RAG_TOP_K
        }
    }


@app.post("/medication/record")
async def record_medication(
    nickname: str,
    medication_name: str,
    notes: Optional[str] = None
):
    """
    복약 완료 기록
    """
    log = medication_reminder.record_medication_taken(nickname, medication_name, notes)
    
    return {
        "status": "success",
        "message": f"{medication_name} 복용이 기록되었습니다.",
        "log": {
            "medication_name": log.medication_name,
            "taken_time": log.taken_time.isoformat() if log.taken_time else None,
            "was_taken": log.was_taken
        }
    }


@app.get("/routine/{nickname}")
async def get_routine_status(nickname: str):
    """
    루틴 상태 조회
    """
    current = routine_manager.get_current_activity(nickname)
    next_activity = routine_manager.get_next_activity(nickname)
    summary = routine_manager.get_daily_summary(nickname)
    suggestions = routine_manager.get_activity_suggestions(nickname)
    
    return {
        "nickname": nickname,
        "current_activity": {
            "activity": current["item"].activity_type.value if current else None,
            "minutes_remaining": current.get("minutes_remaining") if current else None
        } if current else None,
        "next_activity": {
            "activity": next_activity["item"].activity_type.value if next_activity else None,
            "minutes_until": next_activity.get("minutes_until") if next_activity else None
        } if next_activity else None,
        "daily_summary": summary,
        "suggestions": suggestions
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
