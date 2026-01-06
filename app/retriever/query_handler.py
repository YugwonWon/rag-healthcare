"""
RAG 쿼리 핸들러
치매노인 맞춤형 개인화 대화 처리

NER(개체명 인식)과 N-gram 기반 건강 위험 신호 감지 전처리 포함
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
        self._chroma = get_chroma_handler()
        
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
        patient_profile = self._chroma.get_patient_profile(nickname)
        patient_info = self._format_patient_info(patient_profile)
        
        # 2. 관련 문서 검색 (향상된 쿼리 사용)
        doc_results = self._chroma.search_documents(enhanced_query)
        retrieved_context = self._format_retrieved_context(doc_results)
        logger.debug(f"문서 검색 완료 | 결과 수={len(doc_results.get('documents', [[]])[0])}")
        
        # 3. 대화 기록 조회 (개인화) - 최근 3개로 제한하여 응답 속도 개선
        conversation_history = ""
        if include_history:
            conv_results = self._chroma.get_user_conversations(
                nickname=nickname,
                query=query,
                n_results=3
            )
            conversation_history = self._format_conversation_history(conv_results)
        
        # 4. 건강 위험 신호 컨텍스트 추가
        health_context = self._format_health_analysis(health_analysis)
        
        # 현재 한국 시간 가져오기
        current_time = get_kst_datetime_str()
        
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
        self._chroma.add_conversation(
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
        self._logger.info(f"현재 한국 시간: {now.strftime('%Y-%m-%d %H:%M:%S KST')}")
        
        if 5 <= hour < 12:
            time_of_day = "아침"
        elif 12 <= hour < 18:
            time_of_day = "오후"
        else:
            time_of_day = "저녁"
        
        # 최근 활동 조회
        recent_activities = self._chroma.get_recent_activities(nickname, hours=48)
        
        # 개인화된 인사말 생성
        previous_activity_followup = ""
        if recent_activities:
            last_activity = recent_activities[0]
            last_message = last_activity.get("message", "")
            last_time = last_activity.get("timestamp")
            
            # 이전 대화 내용 기반 후속 질문 생성
            if last_time:
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
    
    def _format_retrieved_context(self, results: dict) -> str:
        """검색된 문서 컨텍스트 포맷팅"""
        if not results or not results.get("documents"):
            return "관련 의료 정보 없음"
        
        documents = results["documents"][0] if results["documents"] else []
        
        context_parts = []
        # 컨텍스트 길이 최적화: 각 문서 300자로 제한 (LLM 응답 속도 개선)
        for i, doc in enumerate(documents[:3], 1):  # 최대 3개 문서만 사용
            context_parts.append(f"[{i}] {doc[:300]}")
        
        return "\n\n".join(context_parts)
    
    def _format_conversation_history(self, results: dict) -> str:
        """대화 기록 포맷팅"""
        if not results:
            return "이전 대화 없음"
        
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


def get_query_handler(use_ner_model: bool = True) -> RAGQueryHandler:
    """쿼리 핸들러 인스턴스 가져오기
    
    Args:
        use_ner_model: NER 모델 사용 여부 (False면 더 빠르지만 정확도 낮음)
    """
    return RAGQueryHandler(use_ner_model=use_ner_model)
