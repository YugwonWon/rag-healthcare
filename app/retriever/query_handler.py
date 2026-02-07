"""
RAG 쿼리 핸들러
치매노인 맞춤형 개인화 대화 처리

v2: LangGraph 기반 상태 머신으로 리팩토링
- 의도 분류 → 쿼리 재작성 → 검색(벡터 + GraphRAG) → 응답 생성
- 기존 process_query() 인터페이스 유지 (하위 호환)

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
from app.graph import ConversationState, Intent
from app.graph.graph import get_conversation_graph
from app.logger import get_logger

# LangChain 스토어 (선택적)
if settings.USE_LANGCHAIN_STORE:
    from app.langchain_store import get_langchain_store

logger = get_logger(__name__)


class RAGQueryHandler:
    """RAG 기반 쿼리 처리기
    
    v2: LangGraph 상태 머신 기반
    - 의도 분류 (키워드 기반, LLM 호출 없음)
    - 쿼리 재작성 (후속 질문 맥락 유지)
    - 벡터 검색 + GraphRAG 지식그래프
    - LLM 응답 생성
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
        
        # LangGraph 컴파일된 그래프
        self._graph = get_conversation_graph()
        
        # 전처리 모듈 (기존 호환)
        self._use_ner_model = use_ner_model
        self._health_detector = HealthSignalDetector(use_ner_model=use_ner_model)
    
    async def process_query(
        self,
        nickname: str,
        query: str,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """
        사용자 쿼리 처리 (LangGraph 기반)
        
        기존 인터페이스를 유지하면서 내부적으로 LangGraph 그래프를 실행한다.
        
        Args:
            nickname: 사용자 닉네임
            query: 사용자 질문
            include_history: 대화 기록 포함 여부
        
        Returns:
            Dict containing:
                - response: AI 응답
                - health_analysis: 건강 분석 결과 (선택적)
                - intent: 분류된 의도
                - emergency_alert: 위급 알림 (선택적)
                - graph_context: GraphRAG 컨텍스트 (선택적)
        """
        logger.info(f"쿼리 처리 시작 (LangGraph) | nickname={nickname} | query={query[:50]}...")
        
        # LangGraph 초기 상태 구성
        initial_state: ConversationState = {
            "nickname": nickname,
            "message": query,
        }
        
        # 그래프 실행
        try:
            result = await self._graph.ainvoke(initial_state)
        except Exception as e:
            logger.error(f"LangGraph 실행 오류: {e}", exc_info=True)
            # 폴백: 기본 응답
            return {
                "response": "죄송합니다, 일시적인 오류가 발생했어요. 다시 말씀해 주세요. 🙏",
                "health_analysis": None,
            }
        
        # 결과 추출
        response = result.get("response", "")
        health_analysis = result.get("health_analysis")
        intent = result.get("intent", Intent.GENERAL_CHAT)
        risk_level = result.get("risk_level", "low")
        emergency_alert = result.get("emergency_alert")
        
        logger.info(
            f"쿼리 처리 완료 | intent={intent.value if isinstance(intent, Intent) else intent} "
            f"| risk={risk_level} | response_len={len(response)}"
        )
        
        return {
            "response": response,
            "health_analysis": health_analysis if risk_level != "low" else None,
            "intent": intent.value if isinstance(intent, Intent) else str(intent),
            "emergency_alert": emergency_alert,
            "graph_context": result.get("graph_context", ""),
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
    
    # ==========================================
    # 대화 요약 관련 메서드
    # ==========================================
    
    async def _get_conversation_with_summary(self, nickname: str) -> str:
        """
        요약 + 최근 대화를 결합하여 반환
        
        전략:
        1. 저장된 요약이 있으면 가져옴
        2. 요약 이후의 최근 대화(최대 5개)를 추가
        3. 10번 대화마다 백그라운드에서 요약 갱신
        """
        if not self._use_langchain:
            return ""
        
        try:
            # 저장된 요약 조회
            summary_info = await self._store.get_conversation_summary(nickname)
            
            # 최근 대화 조회 (요약 이후 대화 + 직전 몇개)
            recent_convs = self._store.get_recent_conversations(nickname, limit=5)
            recent_history = self._format_conversation_history(recent_convs)
            
            # 요약이 있으면 결합
            if summary_info and summary_info.get("summary"):
                combined = f"[이전 대화 요약]\n{summary_info['summary']}\n\n[최근 대화]\n{recent_history}"
            else:
                combined = recent_history
            
            # 요약 필요 여부 확인 및 백그라운드 요약 실행
            should_summarize = await self._store.should_summarize(nickname, threshold=10)
            if should_summarize:
                # 비동기로 요약 생성 (응답 블로킹 없음)
                import asyncio
                asyncio.create_task(self._generate_and_save_summary(nickname))
                logger.info(f"백그라운드 요약 시작: {nickname}")
            
            return combined
            
        except Exception as e:
            logger.error(f"대화+요약 조회 오류: {e}")
            # 폴백: 최근 대화만 반환
            recent_convs = self._store.get_recent_conversations(nickname, limit=3)
            return self._format_conversation_history(recent_convs)
    
    async def _generate_and_save_summary(self, nickname: str):
        """
        이전 대화를 요약하여 저장
        
        LLM을 사용해 오래된 대화를 요약하고 DB에 저장
        """
        if not self._use_langchain:
            return
        
        try:
            # 기존 요약 정보 조회
            summary_info = await self._store.get_conversation_summary(nickname)
            start_idx = 0
            if summary_info:
                start_idx = summary_info.get("summarized_count", 0)
            
            # 현재 총 대화 수
            total_count = self._store.get_conversation_count(nickname)
            
            # 요약할 대화 범위 (start_idx ~ total_count - 5)
            # 최근 5개는 요약하지 않고 그대로 유지
            end_idx = max(start_idx, total_count - 5)
            
            if end_idx <= start_idx:
                logger.debug(f"요약할 새 대화 없음: {nickname}")
                return
            
            # 요약할 대화 조회
            conversations_to_summarize = self._store.get_conversations_for_summary(
                nickname, start_idx, end_idx
            )
            
            if not conversations_to_summarize:
                return
            
            # 기존 요약 포함하여 새 요약 생성
            old_summary = summary_info.get("summary", "") if summary_info else ""
            new_summary = await self._summarize_conversations(
                conversations_to_summarize, 
                old_summary
            )
            
            # 요약 저장
            await self._store.save_conversation_summary(
                nickname, 
                new_summary, 
                end_idx
            )
            logger.info(f"대화 요약 완료: {nickname}, {end_idx}개 대화 요약됨")
            
        except Exception as e:
            logger.error(f"요약 생성/저장 오류: {e}")
    
    async def _summarize_conversations(self, conversations: list[dict], old_summary: str = "") -> str:
        """
        LLM을 사용해 대화 내용 요약
        """
        # 대화 포맷팅
        conv_text = ""
        for conv in conversations:
            role = "사용자" if conv["role"] == "user" else "상담사"
            conv_text += f"{role}: {conv['content']}\n"
        
        # 요약 프롬프트
        prompt = f"""다음은 어르신과 건강 상담사의 대화 기록입니다. 
핵심 내용을 3-4문장으로 간결하게 요약해주세요.
어르신의 건강 상태, 주요 고민, 상담 내용을 중심으로 요약합니다.

{f"[이전 요약]{chr(10)}{old_summary}{chr(10)}{chr(10)}" if old_summary else ""}[새 대화]
{conv_text}

요약:"""
        
        try:
            messages = [
                {"role": "system", "content": "당신은 대화 내용을 요약하는 도우미입니다. 간결하고 핵심적인 내용만 요약합니다."},
                {"role": "user", "content": prompt}
            ]
            summary = await self._llm.chat(messages)
            return summary.strip()
        except Exception as e:
            logger.error(f"LLM 요약 오류: {e}")
            # 폴백: 단순 요약
            return f"최근 상담 내용: {len(conversations)//2}회 대화 진행"

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
