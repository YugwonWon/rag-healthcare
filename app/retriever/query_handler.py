"""
RAG 쿼리 핸들러
치매노인 맞춤형 개인화 대화 처리

NER(개체명 인식)과 N-gram 기반 건강 위험 신호 감지 전처리 포함

데이터 레이어:
- USE_LANGCHAIN_STORE=True: LangChain + pgvector (Cloud SQL)
- USE_LANGCHAIN_STORE=False: ChromaDB (기본값)
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from app.config import settings, prompts
from app.model import get_llm
from app.vector_store import get_chroma_handler
from app.utils import get_kst_now, get_kst_datetime_str
from app.preprocessing import (
    HealthSignalDetector,
    KoreanNERProcessor,
    NGramExtractor,
)
from app.preprocessing.health_signal_detector import RiskLevel
from app.logger import get_logger

# LangChain 스토어 (선택적)
if settings.USE_LANGCHAIN_STORE:
    from app.langchain_store import get_langchain_store

logger = get_logger(__name__)


class RAGQueryHandler:
    """RAG 기반 쿼리 처리기
    
    NER + N-gram 전처리로 건강 위험 신호를 감지하고,
    향상된 쿼리로 RAG 검색 수행
    """
    
    def __init__(self, use_ner_model: bool = True):
        """
        Args:
            use_ner_model: NER 모델 사용 여부 (False면 키워드 매칭만 사용)
        """
        self._llm = get_llm()
        
        # 데이터 스토어 선택 (환경변수 기반)
        self._use_langchain = settings.USE_LANGCHAIN_STORE
        if self._use_langchain:
            self._store = get_langchain_store()
            self._chroma = None
            logger.info("LangChain 데이터 스토어 사용 (pgvector)")
        else:
            self._store = None
            self._chroma = get_chroma_handler()
            logger.info("ChromaDB 데이터 스토어 사용")
        
        # 전처리 모듈 초기화
        self._use_ner_model = use_ner_model
        self._health_detector = HealthSignalDetector(use_ner_model=use_ner_model)
    
    async def process_query(
        self,
        nickname: str,
        query: str,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """
        사용자 쿼리 처리 (NER + N-gram 전처리 적용)
        
        Args:
            nickname: 사용자 닉네임
            query: 사용자 질문
            include_history: 대화 기록 포함 여부
        
        Returns:
            Dict containing:
                - response: AI 응답
                - health_analysis: 건강 분석 결과 (선택적)
        """
        logger.info(f"쿼리 처리 시작 | nickname={nickname} | query={query[:50]}...")
        
        # 0. 전처리: NER + N-gram 기반 건강 위험 신호 감지
        health_analysis = self._preprocess_query(query)
        
        # 향상된 쿼리 사용 (건강 용어 + 카테고리 확장)
        enhanced_query = health_analysis.get("enhanced_query", query)
        
        # 건강 위험 수준 확인
        risk_level = health_analysis.get("overall_risk", "low")
        logger.debug(f"전처리 완료 | risk_level={risk_level} | terms={health_analysis.get('detected_health_terms', [])}")
        
        # 1. 환자 프로필 조회
        patient_profile = await self._get_profile(nickname)
        patient_info = self._format_patient_info(patient_profile)
        
        # 2. 관련 문서 검색 (향상된 쿼리 사용)
        doc_results = self._search_documents(enhanced_query)
        retrieved_context = self._format_retrieved_context(doc_results)
        logger.debug(f"문서 검색 완료 | 결과 수={len(doc_results) if isinstance(doc_results, list) else len(doc_results.get('documents', [[]])[0])}")
        
        # 3. 대화 기록 조회 (개인화) - 최근 3개로 제한하여 응답 속도 개선
        conversation_history = ""
        activity_context = ""
        if include_history:
            conv_results = self._get_conversations(nickname, query, n_results=3)
            conversation_history = self._format_conversation_history(conv_results)
            
            # 최근 활동 요약 추가 (ChromaDB 전용, LangChain은 추후 지원)
            if not self._use_langchain:
                activity_summary = self._chroma.get_user_activity_summary(nickname, hours=24)
                activity_context = self._format_activity_summary(activity_summary)
        
        # 4. 건강 위험 신호 컨텍스트 추가
        health_context = self._format_health_analysis(health_analysis)
        
        # 현재 한국 시간 가져오기
        current_time = get_kst_datetime_str()
        
        # 대화 기록에 활동 요약 추가
        if activity_context:
            conversation_history = f"{conversation_history}\n\n{activity_context}"
        
        # 5. 프롬프트 구성 (건강 분석 결과 포함)
        system_prompt = prompts.SYSTEM_PROMPT.format(
            current_time=current_time,
            patient_info=patient_info,
            conversation_history=conversation_history,
            retrieved_context=retrieved_context
        )
        
        # 건강 위험이 감지되면 추가 지시사항 포함
        if risk_level in ["high", "critical"]:
            logger.warning(f"건강 위험 감지 | nickname={nickname} | risk_level={risk_level}")
            system_prompt += f"\n\n[건강 위험 감지]\n{health_context}"
            system_prompt += "\n주의: 사용자의 건강 상태에 주의를 기울이고, 필요시 보호자 연락이나 전문가 상담을 안내하세요."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # 6. LLM 응답 생성
        logger.debug("LLM 응답 생성 중...")
        response = await self._llm.chat(messages)
        logger.info(f"LLM 응답 완료 | 길이={len(response)}")
        
        # 7. 대화 기록 저장 (건강 분석 메타데이터 포함)
        self._save_conversation(
            nickname=nickname,
            user_message=query,
            assistant_response=response,
            metadata={
                "health_terms": health_analysis.get("detected_health_terms", [])[:5],
                "risk_level": risk_level,
                "risk_categories": [r["category"] for r in health_analysis.get("risk_categories", [])]
            }
        )
        
        return {
            "response": response,
            "health_analysis": health_analysis if risk_level != "low" else None
        }
    
    # ==========================================
    # 데이터 레이어 추상화 메서드
    # ==========================================
    
    async def _get_profile(self, nickname: str) -> dict:
        """프로필 조회 (LangChain/ChromaDB 분기)"""
        if self._use_langchain:
            return await self._store.get_profile(nickname)
        else:
            return self._chroma.get_patient_profile(nickname)
    
    def _search_documents(self, query: str, k: int = 5) -> Any:
        """문서 검색 (LangChain/ChromaDB 분기)"""
        if self._use_langchain:
            return self._store.search_documents(query, k=k)
        else:
            return self._chroma.search_documents(query)
    
    def _get_conversations(self, nickname: str, query: str, n_results: int = 3) -> Any:
        """대화 기록 조회 (LangChain/ChromaDB 분기)"""
        if self._use_langchain:
            return self._store.get_recent_conversations(nickname, limit=n_results)
        else:
            return self._chroma.get_user_conversations(
                nickname=nickname,
                query=query,
                n_results=n_results
            )
    
    def _save_conversation(self, nickname: str, user_message: str, 
                          assistant_response: str, metadata: dict = None):
        """대화 저장 (LangChain/ChromaDB 분기)"""
        if self._use_langchain:
            self._store.save_conversation(nickname, user_message, assistant_response)
        else:
            self._chroma.add_conversation(
                nickname=nickname,
                user_message=user_message,
                assistant_response=assistant_response,
                metadata=metadata
            )
    
    def _preprocess_query(self, query: str) -> Dict[str, Any]:
        """NER + N-gram 기반 쿼리 전처리
        
        논문 방법론:
        1. NER로 건강 관련 용어 태깅
        2. 태깅된 용어 전후 5단어 N-gram 추출
        3. 규칙 기반 건강 위험 신호 감지
        4. 향상된 쿼리 생성
        """
        try:
            result = self._health_detector.get_risk_summary(query)
            logger.debug(f"건강 분석 결과: {result}")
            return result
        except Exception as e:
            logger.warning(f"전처리 오류 (기본값 사용): {e}")
            return {
                "overall_risk": "low",
                "detected_health_terms": [],
                "risk_categories": [],
                "summary": "",
                "enhanced_query": query
            }
    
    def _format_health_analysis(self, analysis: Dict[str, Any]) -> str:
        """건강 분석 결과 포맷팅"""
        if not analysis or analysis.get("overall_risk") == "low":
            return ""
        
        parts = []
        
        # 감지된 건강 용어
        terms = analysis.get("detected_health_terms", [])
        if terms:
            parts.append(f"감지된 건강 관련 용어: {', '.join(terms[:5])}")
        
        # 위험 카테고리
        categories = analysis.get("risk_categories", [])
        for cat in categories[:3]:
            parts.append(f"- {cat.get('category', '')}: {cat.get('description', '')}")
        
        # 요약
        summary = analysis.get("summary", "")
        if summary:
            parts.append(f"\n분석 요약: {summary}")
        
        return "\n".join(parts)
    
    async def generate_greeting(self, nickname: str) -> str:
        """
        개인화된 인사말 생성
        
        Args:
            nickname: 사용자 닉네임
        
        Returns:
            개인화된 인사말
        """
        # 현재 한국 시간 확인
        now = get_kst_now()
        hour = now.hour
        logger.info(f"현재 한국 시간: {now.strftime('%Y-%m-%d %H:%M:%S KST')}")
        
        if 5 <= hour < 12:
            time_of_day = "아침"
        elif 12 <= hour < 18:
            time_of_day = "오후"
        else:
            time_of_day = "저녁"
        
        # 최근 활동 조회 (ChromaDB 사용 시에만)
        recent_activities = []
        if self._chroma:
            recent_activities = self._chroma.get_recent_activities(nickname, hours=48)
        
        # 개인화된 인사말 생성
        previous_activity_followup = ""
        if recent_activities:
            last_activity = recent_activities[0]
            last_message = last_activity.get("message", "")
            last_time = last_activity.get("timestamp")
            
            # 이전 대화 내용 기반 후속 질문 생성
            if last_time:
                # timezone-naive인 경우 KST로 변환
                if last_time.tzinfo is None:
                    from app.healthcare.daily_routine import KST
                    last_time = last_time.replace(tzinfo=KST)
                time_diff = now - last_time
                hours_ago = time_diff.total_seconds() / 3600
                
                # 키워드 기반 후속 질문
                followup_prompts = self._generate_followup_prompt(last_message, hours_ago)
                if followup_prompts:
                    previous_activity_followup = followup_prompts
        
        greeting = prompts.DAILY_CHECK_IN.format(
            nickname=nickname,
            time_of_day=time_of_day,
            previous_activity_followup=previous_activity_followup
        )
        
        return greeting
    
    def _generate_followup_prompt(self, last_message: str, hours_ago: float) -> str:
        """이전 대화 기반 후속 질문 생성"""
        keywords_map = {
            "산책": "산책 다녀오셨나요? 날씨가 좋았나요?",
            "약": "약은 잘 드셨나요?",
            "밥": "식사는 맛있게 하셨나요?",
            "식사": "식사는 맛있게 하셨나요?",
            "잠": "푹 주무셨나요?",
            "운동": "운동은 잘 하셨나요?",
            "병원": "병원 다녀오셨나요? 어떠셨어요?",
            "가족": "가족분들과 좋은 시간 보내셨나요?",
            "TV": "재미있는 프로그램 보셨나요?",
            "음악": "좋은 음악 들으셨나요?",
        }
        
        for keyword, followup in keywords_map.items():
            if keyword in last_message.lower():
                if hours_ago < 24:
                    return f"어제 {keyword} 이야기 하셨는데, {followup}"
                elif hours_ago < 48:
                    return f"그저께 {keyword} 말씀하셨는데, {followup}"
        
        return ""
    
    def _format_patient_info(self, profile: Optional[dict]) -> str:
        """환자 정보 포맷팅"""
        if not profile:
            return "등록된 환자 정보가 없습니다."
        
        info_lines = []
        for key, value in profile.items():
            if key != "nickname":
                info_lines.append(f"- {key}: {value}")
        
        return "\n".join(info_lines) if info_lines else "기본 정보만 등록됨"
    
    def _format_retrieved_context(self, results: Any) -> str:
        """검색된 문서 컨텍스트 포맷팅 (LangChain/ChromaDB 분기)"""
        if not results:
            return "관련 의료 정보 없음"
        
        context_parts = []
        
        # LangChain 형식: list[dict] with "content", "score"
        if self._use_langchain and isinstance(results, list):
            for i, doc in enumerate(results[:3], 1):
                content = doc.get("content", "")[:300]
                context_parts.append(f"[{i}] {content}")
        # ChromaDB 형식: dict with "documents"
        elif isinstance(results, dict) and results.get("documents"):
            documents = results["documents"][0] if results["documents"] else []
            for i, doc in enumerate(documents[:3], 1):
                context_parts.append(f"[{i}] {doc[:300]}")
        
        return "\n\n".join(context_parts) if context_parts else "관련 의료 정보 없음"
    
    def _format_conversation_history(self, results: Any) -> str:
        """대화 기록 포맷팅 (LangChain/ChromaDB 분기)"""
        if not results:
            return "이전 대화 없음"
        
        history_parts = []
        
        # LangChain 형식: list[dict] with "role", "content"
        if self._use_langchain and isinstance(results, list):
            for msg in results[-6:]:  # 최근 3쌍 (6개 메시지)
                role = "사용자" if msg.get("role") == "user" else "AI"
                content = msg.get("content", "")[:200]
                history_parts.append(f"{role}: {content}")
            return "\n".join(history_parts) if history_parts else "이전 대화 없음"
        
        # ChromaDB 형식: dict with "documents"
        documents = results.get("documents", [])
        if isinstance(documents, list) and len(documents) > 0:
            if isinstance(documents[0], list):
                documents = documents[0]
        
        if not documents:
            return "이전 대화 없음"
        
        history_parts = []
        for doc in documents[:3]:  # 최근 3개 대화만 포함 (응답 속도 최적화)
            # 각 대화도 200자로 제한
            history_parts.append(doc[:200] if len(doc) > 200 else doc)
        
        return "\n---\n".join(history_parts)
    
    def _format_activity_summary(self, activity_data: dict) -> str:
        """활동 요약 포맷팅"""
        if not activity_data:
            return ""
        
        summary = activity_data.get("summary", {})
        if not summary:
            return ""
        
        from app.healthcare.daily_routine import get_kst_now
        
        parts = ["[오늘의 활동 기록]"]
        for activity, info in summary.items():
            count = info.get("count", 0)
            last_time = info.get("last_time")
            if last_time:
                # 시간 차이 계산
                now = get_kst_now()
                if last_time.tzinfo is None:
                    from app.healthcare.daily_routine import KST
                    last_time = last_time.replace(tzinfo=KST)
                diff = now - last_time
                hours_ago = int(diff.total_seconds() / 3600)
                if hours_ago < 1:
                    time_str = "방금 전"
                elif hours_ago < 24:
                    time_str = f"{hours_ago}시간 전"
                else:
                    time_str = f"{hours_ago // 24}일 전"
                parts.append(f"- {activity}: {count}회 (마지막: {time_str})")
            else:
                parts.append(f"- {activity}: {count}회")
        
        return "\n".join(parts) if len(parts) > 1 else ""


def get_query_handler(use_ner_model: bool = True) -> RAGQueryHandler:
    """쿼리 핸들러 인스턴스 가져오기
    
    Args:
        use_ner_model: NER 모델 사용 여부 (False면 더 빠르지만 정확도 낮음)
    """
    return RAGQueryHandler(use_ner_model=use_ner_model)
