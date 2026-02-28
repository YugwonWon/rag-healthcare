"""
건강 위험 신호 감지 모듈
바른 형태소 분석기 기반 의료 개체 인식 + N-gram + 규칙 기반 건강 위험 감지

방법론:
1. bareunpy 형태소 분석 → 어근 복원 + 불용어 제거
2. 의료 용어 사전 매칭 (어근 기준, 카테고리 분류)
3. 복합 패턴 감지 + N-gram 컨텍스트 추출
4. 규칙 기반 위험 수준 판정
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from .medical_entity import (
    BareunMorphAnalyzer, MorphAnalysisResult, MedicalEntity,
    MedicalCategory, get_morph_analyzer,
)
from .ngram_extractor import NGramExtractor, NGramResult
from app.logger import get_logger

logger = get_logger(__name__)


class RiskLevel(Enum):
    """건강 위험 수준"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(Enum):
    """건강 위험 카테고리 (논문 기반)"""
    CHRONIC_DISEASE = "chronic_disease"  # 만성질환
    SLEEP_DISORDER = "sleep_disorder"  # 수면 장애
    FALL_RISK = "fall_risk"  # 낙상 위험
    NUTRITIONAL = "nutritional"  # 영양 문제
    PAIN = "pain"  # 통증 관리
    EMOTIONAL = "emotional"  # 정서적 웰빙
    COGNITIVE = "cognitive"  # 인지 기능
    MEDICATION = "medication"  # 약물 관련
    HYGIENE = "hygiene"  # 위생/일상생활
    EMERGENCY = "emergency"  # 응급 상황


@dataclass
class HealthRiskSignal:
    """감지된 건강 위험 신호"""
    category: RiskCategory
    level: RiskLevel
    trigger_terms: List[str]  # 감지를 유발한 용어들
    context: str  # N-gram 컨텍스트
    description: str  # 위험 설명
    recommendation: str  # 권장 조치


@dataclass
class HealthAnalysisResult:
    """건강 분석 결과"""
    original_text: str
    morph_result: Optional[MorphAnalysisResult] = None
    ngram_result: Optional[NGramResult] = None
    risk_signals: List[HealthRiskSignal] = field(default_factory=list)
    overall_risk_level: RiskLevel = RiskLevel.LOW
    summary: str = ""
    enhanced_query: str = ""  # RAG 검색을 위해 향상된 쿼리


class HealthSignalDetector:
    """건강 위험 신호 감지기
    
    bareunpy 형태소 분석 기반 의료 개체 인식 + 규칙 기반 위험 감지:
    1. 형태소 분석으로 어근 복원 후 의료 용어 사전 매칭
    2. 매칭된 용어 전후 5단어 N-gram 컨텍스트 추출
    3. 규칙 기반 위험 신호 감지 + 컨텍스트 패턴으로 수준 조정
    """
    
    # 카테고리별 키워드 (논문의 health signal keywords 기반)
    RISK_KEYWORDS = {
        RiskCategory.CHRONIC_DISEASE: {
            "keywords": ["당뇨", "혈당", "혈압", "고혈압", "저혈압", "심장", "콜레스테롤", 
                        "관절염", "골다공증", "신장", "간", "폐", "천식", "암"],
            "level": RiskLevel.HIGH,
            "description": "만성질환 관련 증상이 감지되었습니다.",
            "recommendation": "정기적인 건강 검진과 의사 상담을 권장합니다."
        },
        RiskCategory.SLEEP_DISORDER: {
            "keywords": ["불면", "잠", "수면", "새벽", "뒤척", "수면제", "악몽", "낮잠",
                        "졸음", "피곤", "피로", "기상", "수면무호흡"],
            "level": RiskLevel.MEDIUM,
            "description": "수면 관련 문제가 감지되었습니다.",
            "recommendation": "규칙적인 수면 습관과 수면 환경 개선을 권장합니다."
        },
        RiskCategory.FALL_RISK: {
            "keywords": ["넘어", "낙상", "미끄러", "쓰러", "어지럽", "현기증", "균형",
                        "다리", "걸음", "비틀", "골절", "멍"],
            "level": RiskLevel.HIGH,
            "description": "낙상 위험 신호가 감지되었습니다.",
            "recommendation": "이동 시 주의하시고, 필요시 보조 기구 사용을 고려하세요."
        },
        RiskCategory.NUTRITIONAL: {
            "keywords": ["식욕", "입맛", "먹다", "밥", "식사", "체중", "살", "영양",
                        "구토", "소화", "체하다", "탈수", "변비", "설사"],
            "level": RiskLevel.MEDIUM,
            "description": "영양 및 식이 관련 문제가 감지되었습니다.",
            "recommendation": "균형 잡힌 식단과 충분한 수분 섭취를 권장합니다."
        },
        RiskCategory.PAIN: {
            "keywords": ["아프", "통증", "쑤시", "결리", "저리", "두통", "복통",
                        "허리", "무릎", "어깨", "목", "관절통", "신경통"],
            "level": RiskLevel.MEDIUM,
            "description": "통증 관련 증상이 감지되었습니다.",
            "recommendation": "통증이 지속되면 전문의 상담을 권장합니다."
        },
        RiskCategory.EMOTIONAL: {
            "keywords": ["우울", "슬프", "불안", "걱정", "외롭", "무기력", "스트레스",
                        "화나", "짜증", "눈물", "절망", "두렵", "무섭"],
            "level": RiskLevel.MEDIUM,
            "description": "정서적 어려움이 감지되었습니다.",
            "recommendation": "가까운 사람과 대화하거나 전문 상담을 받아보세요."
        },
        RiskCategory.COGNITIVE: {
            "keywords": ["기억", "잊어", "깜빡", "헷갈", "혼란", "치매", "건망증",
                        "집중", "인지", "판단", "방향", "길"],
            "level": RiskLevel.HIGH,
            "description": "인지 기능 관련 증상이 감지되었습니다.",
            "recommendation": "인지 기능 평가와 전문의 상담을 권장합니다."
        },
        RiskCategory.MEDICATION: {
            "keywords": ["약", "복용", "처방", "부작용", "약물", "주사", "치료",
                        "복약", "투약", "알레르기", "과량"],
            "level": RiskLevel.MEDIUM,
            "description": "약물 관련 문제가 감지되었습니다.",
            "recommendation": "약 복용 시간과 용량을 확인하고, 이상 반응시 의사와 상담하세요."
        },
        RiskCategory.HYGIENE: {
            "keywords": ["목욕", "샤워", "세수", "양치", "옷", "청소", "위생",
                        "화장실", "소변", "대변", "기저귀"],
            "level": RiskLevel.LOW,
            "description": "일상 위생 관련 사항이 감지되었습니다.",
            "recommendation": "규칙적인 위생 관리 루틴을 유지하세요."
        },
        RiskCategory.EMERGENCY: {
            "keywords": ["가슴", "호흡", "숨", "의식", "경련", "마비", "출혈",
                        "응급", "119", "구급"],
            "level": RiskLevel.CRITICAL,
            "description": "응급 상황 가능성이 감지되었습니다!",
            "recommendation": "즉시 119에 연락하거나 보호자에게 알리세요."
        }
    }
    
    # 부정적 컨텍스트 패턴 (위험 수준 높임)
    NEGATIVE_PATTERNS = [
        r"안\s*(좋|되|나아|먹|자|움직)",  # 안 좋다, 안 되다
        r"못\s*(자|먹|움직|걷)",  # 못 자다
        r"(많이|자꾸|계속|심하게)\s*",  # 강조 표현
        r"(어제|오늘|요즘|최근)\s*(부터|들어)",  # 시간 표현
        r"\d+\s*(일|번|회|시간)",  # 빈도 표현
    ]
    
    # 긍정적 컨텍스트 패턴 (위험 수준 낮춤)
    POSITIVE_PATTERNS = [
        r"(좋아|나아|괜찮|편해|잘)",  # 긍정 표현
        r"(없|안)\s*(아프|통증|문제)",  # 부정 + 증상 = 긍정
    ]
    
    def __init__(
        self,
        morph_analyzer: Optional[BareunMorphAnalyzer] = None,
        ngram_extractor: Optional[NGramExtractor] = None,
        use_ner_model: bool = True,  # 하위 호환 (무시됨)
        embedding_model = None,
    ):
        """
        Args:
            morph_analyzer: 바른 형태소 분석기 (None이면 싱글톤 사용)
            ngram_extractor: N-gram 추출기 (None이면 기본값 사용)
            use_ner_model: 하위 호환용 (무시됨, 기존 NER은 제거)
            embedding_model: 의미적 유사도 분석용 임베딩 모델
        """
        self.morph_analyzer = morph_analyzer or get_morph_analyzer()
        self.ngram_extractor = ngram_extractor or NGramExtractor()
        self.embedding_model = embedding_model
        
    def _detect_risk_by_keywords(
        self, 
        text: str, 
        health_terms: List[str]
    ) -> List[HealthRiskSignal]:
        """키워드 기반 위험 신호 감지"""
        signals = []
        text_lower = text.lower()
        detected_categories = set()
        
        for category, config in self.RISK_KEYWORDS.items():
            keywords = config["keywords"]
            matched_keywords = []
            
            # 텍스트에서 키워드 매칭
            for keyword in keywords:
                if keyword in text_lower:
                    matched_keywords.append(keyword)
                    
            # health_terms에서도 매칭
            for term in health_terms:
                for keyword in keywords:
                    if keyword in term.lower() or term.lower() in keyword:
                        if term not in matched_keywords:
                            matched_keywords.append(term)
            
            if matched_keywords and category not in detected_categories:
                detected_categories.add(category)
                
                # 컨텍스트 분석으로 위험 수준 조정
                risk_level = config["level"]
                risk_level = self._adjust_risk_level(text, risk_level)
                
                signal = HealthRiskSignal(
                    category=category,
                    level=risk_level,
                    trigger_terms=matched_keywords,
                    context=text[:200],  # 컨텍스트 200자 제한
                    description=config["description"],
                    recommendation=config["recommendation"]
                )
                signals.append(signal)
                
        return signals
    
    def _adjust_risk_level(self, text: str, base_level: RiskLevel) -> RiskLevel:
        """컨텍스트 패턴으로 위험 수준 조정"""
        # 부정적 패턴 확인
        negative_count = 0
        for pattern in self.NEGATIVE_PATTERNS:
            if re.search(pattern, text):
                negative_count += 1
                
        # 긍정적 패턴 확인
        positive_count = 0
        for pattern in self.POSITIVE_PATTERNS:
            if re.search(pattern, text):
                positive_count += 1
        
        # 수준 조정
        level_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        current_idx = level_order.index(base_level)
        
        # 부정적 패턴이 많으면 위험 수준 상승
        if negative_count >= 2 and current_idx < len(level_order) - 1:
            return level_order[current_idx + 1]
            
        # 긍정적 패턴이 있으면 위험 수준 하강
        if positive_count >= 1 and current_idx > 0:
            return level_order[current_idx - 1]
            
        return base_level
    
    def _calculate_overall_risk(
        self, 
        signals: List[HealthRiskSignal]
    ) -> RiskLevel:
        """전체 위험 수준 계산"""
        if not signals:
            return RiskLevel.LOW
            
        # 가장 높은 위험 수준 반환
        level_priority = {
            RiskLevel.CRITICAL: 4,
            RiskLevel.HIGH: 3,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 1
        }
        
        max_level = max(signals, key=lambda s: level_priority[s.level])
        return max_level.level
    
    def _generate_summary(
        self, 
        signals: List[HealthRiskSignal],
        overall_level: RiskLevel
    ) -> str:
        """분석 요약 생성"""
        if not signals:
            return "건강 관련 특이 사항이 감지되지 않았습니다."
            
        level_names = {
            RiskLevel.LOW: "낮음",
            RiskLevel.MEDIUM: "중간",
            RiskLevel.HIGH: "높음",
            RiskLevel.CRITICAL: "긴급"
        }
        
        category_names = {
            RiskCategory.CHRONIC_DISEASE: "만성질환",
            RiskCategory.SLEEP_DISORDER: "수면",
            RiskCategory.FALL_RISK: "낙상 위험",
            RiskCategory.NUTRITIONAL: "영양",
            RiskCategory.PAIN: "통증",
            RiskCategory.EMOTIONAL: "정서",
            RiskCategory.COGNITIVE: "인지",
            RiskCategory.MEDICATION: "약물",
            RiskCategory.HYGIENE: "위생",
            RiskCategory.EMERGENCY: "응급"
        }
        
        categories = [category_names.get(s.category, str(s.category)) for s in signals]
        
        summary = f"전체 위험 수준: {level_names[overall_level]}\n"
        summary += f"감지된 영역: {', '.join(set(categories))}\n"
        
        if overall_level == RiskLevel.CRITICAL:
            summary += "\n⚠️ 긴급 상황이 감지되었습니다. 즉시 조치가 필요합니다."
        
        return summary
    
    def _generate_enhanced_query(
        self, 
        original_text: str,
        health_terms: List[str],
        signals: List[HealthRiskSignal]
    ) -> str:
        """RAG 검색을 위한 향상된 쿼리 생성"""
        # 건강 용어와 카테고리 기반 쿼리 확장
        query_parts = [original_text]
        
        # 주요 건강 용어 추가
        if health_terms:
            query_parts.append(" ".join(health_terms[:5]))
        
        # 감지된 위험 카테고리 관련 용어 추가
        category_terms = {
            RiskCategory.CHRONIC_DISEASE: "만성질환 관리 치료",
            RiskCategory.SLEEP_DISORDER: "수면 개선 불면증",
            RiskCategory.FALL_RISK: "낙상 예방 안전",
            RiskCategory.NUTRITIONAL: "영양 식이 관리",
            RiskCategory.PAIN: "통증 완화 관리",
            RiskCategory.EMOTIONAL: "정서 지원 우울",
            RiskCategory.COGNITIVE: "인지 훈련 치매 예방",
            RiskCategory.MEDICATION: "약물 복용 관리",
            RiskCategory.HYGIENE: "일상생활 지원",
            RiskCategory.EMERGENCY: "응급 처치 대응"
        }
        
        for signal in signals[:3]:  # 상위 3개 카테고리만
            if signal.category in category_terms:
                query_parts.append(category_terms[signal.category])
        
        return " ".join(query_parts)
    
    def analyze(self, text: str) -> HealthAnalysisResult:
        """텍스트 건강 위험 분석 수행
        
        Args:
            text: 분석할 대화 텍스트
            
        Returns:
            HealthAnalysisResult: 종합 분석 결과
        """
        result = HealthAnalysisResult(original_text=text)
        
        # 1. 형태소 분석 + 의료 개체 인식 (bareunpy 기반)
        morph_result = self.morph_analyzer.analyze(text)
        result.morph_result = morph_result
        
        # 건강 용어 리스트 (중복 제거)
        health_terms = list(set(e.lemma for e in morph_result.medical_entities))
        health_term_positions = [(e.text, e.start, e.end) for e in morph_result.medical_entities]
        
        # 2. N-gram 컨텍스트 추출
        if health_term_positions:
            ngram_result = self.ngram_extractor.extract_all_contexts(
                text, health_term_positions
            )
            result.ngram_result = ngram_result
        
        # 3. 규칙 기반 위험 신호 감지
        risk_signals = self._detect_risk_by_keywords(text, health_terms)
        result.risk_signals = risk_signals
        
        # 4. 전체 위험 수준 계산
        result.overall_risk_level = self._calculate_overall_risk(risk_signals)
        
        # 5. 요약 생성
        result.summary = self._generate_summary(
            risk_signals, result.overall_risk_level
        )
        
        # 6. 향상된 쿼리 생성
        result.enhanced_query = self._generate_enhanced_query(
            text, health_terms, risk_signals
        )
        
        return result
    
    def get_risk_summary(self, text: str) -> Dict[str, Any]:
        """간단한 위험 요약 반환 (API용)"""
        result = self.analyze(text)
        
        return {
            "overall_risk": result.overall_risk_level.value,
            "detected_health_terms": [e.lemma for e in result.morph_result.medical_entities][:10] if result.morph_result else [],
            "risk_categories": [
                {
                    "category": s.category.value,
                    "level": s.level.value,
                    "description": s.description
                }
                for s in result.risk_signals
            ],
            "summary": result.summary,
            "enhanced_query": result.enhanced_query
        }


# 간편 함수
def detect_health_signals(text: str, **kwargs) -> HealthAnalysisResult:
    """건강 위험 신호 감지 (간편 인터페이스)"""
    detector = HealthSignalDetector()
    return detector.analyze(text)


def preprocess_for_rag(text: str, **kwargs) -> str:
    """RAG 검색을 위한 전처리 (향상된 쿼리 반환)"""
    result = detect_health_signals(text)
    return result.enhanced_query
