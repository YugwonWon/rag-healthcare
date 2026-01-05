"""
한국어 개체명 인식(NER) 모듈
KoELECTRA 기반 NER 모델을 사용하여 건강 관련 개체명 추출
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from app.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NamedEntity:
    """개체명 정보를 담는 데이터 클래스"""
    text: str  # 인식된 텍스트
    label: str  # 개체명 레이블 (예: TERM, LOCATION 등)
    start: int  # 시작 위치
    end: int  # 끝 위치
    score: float  # 신뢰도 점수
    

@dataclass
class NERResult:
    """NER 분석 결과"""
    original_text: str
    entities: List[NamedEntity] = field(default_factory=list)
    health_entities: List[NamedEntity] = field(default_factory=list)  # 건강 관련 개체
    

class KoreanNERProcessor:
    """한국어 NER 처리기
    
    KoELECTRA-small-v3-modu-ner 모델을 사용하여
    한국어 텍스트에서 개체명을 인식하고, 건강 관련 개체를 추출합니다.
    
    태그셋 (TTA 대분류 15가지):
    - ARTIFACTS (AF): 인공물 (문화재, 건물 등)
    - ANIMAL (AM): 동물
    - CIVILIZATION (CV): 문명/문화
    - DATE (DT): 날짜/시기
    - EVENT (EV): 사건/행사
    - STUDY_FIELD (FD): 학문 분야
    - LOCATION (LC): 지역/장소
    - MATERIAL (MT): 물질
    - ORGANIZATION (OG): 기관/단체
    - PERSON (PS): 인명
    - PLANT (PT): 식물
    - QUANTITY (QT): 수량
    - TIME (TI): 시간
    - TERM (TM): 일반 용어
    - THEORY (TR): 이론/법칙
    """
    
    # 건강 관련 키워드 (논문 기반)
    HEALTH_KEYWORDS = {
        # 병원/의료 관련
        "병원", "의원", "클리닉", "진료", "진찰", "치료", "수술", "입원", "퇴원",
        "의사", "간호사", "약사", "한의원", "응급실", "외래", "내과", "정형외과",
        # 수면 관련
        "잠", "수면", "불면", "불면증", "숙면", "낮잠", "밤잠", "새벽", "기상",
        "잠들다", "깨다", "뒤척이다", "수면제",
        # 배뇨/배변 관련
        "소변", "대변", "배변", "화장실", "변비", "설사", "요실금", "빈뇨",
        # 낙상 관련
        "낙상", "넘어지다", "미끄러지다", "쓰러지다", "다치다", "멍", "골절",
        "어지럽다", "어지러움", "현기증", "균형",
        # 통증 관련
        "통증", "아프다", "쑤시다", "결리다", "저리다", "두통", "복통", 
        "관절통", "허리", "무릎", "어깨", "목", "손목", "발목",
        # 식이/영양 관련
        "식사", "밥", "음식", "식욕", "입맛", "먹다", "영양", "체중", 
        "살", "비만", "저체중", "당뇨", "혈당",
        # 약 관련
        "약", "복용", "처방", "부작용", "알약", "물약", "연고", "주사",
        # 감정/정서 관련
        "우울", "불안", "걱정", "스트레스", "외롭다", "슬프다", "화나다",
        "무기력", "의욕", "기분", "감정",
        # 인지 관련
        "기억", "건망증", "치매", "인지", "집중", "혼란", "헷갈리다",
        "잊어버리다", "깜빡하다",
        # 일상 활동 관련
        "산책", "운동", "걷기", "체조", "스트레칭", "목욕", "샤워", "세수",
        "양치", "옷", "착용", "외출",
        # 신체 증상
        "열", "기침", "콧물", "가래", "호흡", "숨", "심장", "혈압", 
        "맥박", "체온", "부종", "붓다"
    }
    
    # 건강 관련 NER 레이블 (일반적으로 건강 정보가 포함될 수 있는 레이블)
    HEALTH_RELATED_LABELS = {"TERM", "DATE", "TIME", "QUANTITY", "LOCATION"}
    
    def __init__(
        self,
        model_name: str = "Leo97/KoELECTRA-small-v3-modu-ner",
        confidence_threshold: float = 0.5,
        use_gpu: bool = False
    ):
        """
        Args:
            model_name: HuggingFace 모델 이름
            confidence_threshold: 개체 인식 신뢰도 임계값
            use_gpu: GPU 사용 여부
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        self._pipeline = None
        self._is_loaded = False
        
    def load_model(self) -> bool:
        """NER 모델 로드"""
        if self._is_loaded:
            return True
            
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            
            logger.info(f"NER 모델 로딩 중: {self.model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            
            device = 0 if self.use_gpu else -1
            self._pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",  # BIO 태그 병합
                device=device
            )
            
            self._is_loaded = True
            logger.info("NER 모델 로딩 완료")
            return True
            
        except ImportError as e:
            logger.error(f"transformers 라이브러리가 필요합니다: {e}")
            return False
        except Exception as e:
            logger.error(f"NER 모델 로딩 실패: {e}")
            return False
    
    def _extract_entities(self, text: str) -> List[NamedEntity]:
        """텍스트에서 개체명 추출"""
        if not self._is_loaded:
            if not self.load_model():
                return []
        
        try:
            results = self._pipeline(text)
            entities = []
            
            for item in results:
                if item.get("score", 0) >= self.confidence_threshold:
                    entity = NamedEntity(
                        text=item.get("word", "").replace("##", ""),  # subword 토큰 처리
                        label=item.get("entity_group", "O"),
                        start=item.get("start", 0),
                        end=item.get("end", 0),
                        score=item.get("score", 0.0)
                    )
                    entities.append(entity)
                    
            return entities
            
        except Exception as e:
            logger.error(f"개체명 추출 오류: {e}")
            return []
    
    def _is_health_related(self, entity: NamedEntity) -> bool:
        """개체가 건강 관련인지 판단"""
        # 건강 키워드와 매칭
        entity_text = entity.text.lower()
        for keyword in self.HEALTH_KEYWORDS:
            if keyword in entity_text or entity_text in keyword:
                return True
        
        # 건강 관련 레이블인 경우 추가 검사
        if entity.label in self.HEALTH_RELATED_LABELS:
            # 숫자가 포함된 QUANTITY는 건강 측정값일 수 있음
            if entity.label == "QUANTITY":
                return True
            # TIME/DATE는 증상 발생 시간일 수 있음
            if entity.label in {"TIME", "DATE"}:
                return True
                
        return False
    
    def _find_health_keywords_direct(self, text: str) -> List[NamedEntity]:
        """NER 모델 없이 직접 건강 키워드 매칭"""
        entities = []
        text_lower = text.lower()
        
        for keyword in self.HEALTH_KEYWORDS:
            # 키워드의 모든 출현 위치 찾기
            start = 0
            while True:
                pos = text_lower.find(keyword, start)
                if pos == -1:
                    break
                    
                entity = NamedEntity(
                    text=text[pos:pos + len(keyword)],
                    label="HEALTH_TERM",
                    start=pos,
                    end=pos + len(keyword),
                    score=1.0  # 직접 매칭은 신뢰도 1.0
                )
                entities.append(entity)
                start = pos + 1
                
        # 중복 제거 및 정렬
        entities = self._remove_overlapping_entities(entities)
        return entities
    
    def _remove_overlapping_entities(self, entities: List[NamedEntity]) -> List[NamedEntity]:
        """겹치는 개체 중 더 긴 것만 유지"""
        if not entities:
            return []
            
        # 시작 위치로 정렬
        sorted_entities = sorted(entities, key=lambda x: (x.start, -(x.end - x.start)))
        result = []
        
        for entity in sorted_entities:
            # 이전 개체와 겹치지 않으면 추가
            if not result or entity.start >= result[-1].end:
                result.append(entity)
            # 현재 개체가 더 길면 교체
            elif (entity.end - entity.start) > (result[-1].end - result[-1].start):
                result[-1] = entity
                
        return result
    
    def process(self, text: str, use_model: bool = True) -> NERResult:
        """텍스트에서 NER 수행 및 건강 관련 개체 추출
        
        Args:
            text: 분석할 텍스트
            use_model: True면 NER 모델 사용, False면 키워드 매칭만 사용
            
        Returns:
            NERResult: NER 분석 결과
        """
        result = NERResult(original_text=text)
        
        if use_model:
            # NER 모델로 개체 추출
            entities = self._extract_entities(text)
            result.entities = entities
            
            # 건강 관련 개체 필터링
            result.health_entities = [e for e in entities if self._is_health_related(e)]
            
            # NER로 찾지 못한 건강 키워드도 직접 찾기
            keyword_entities = self._find_health_keywords_direct(text)
            
            # 기존 개체와 겹치지 않는 것만 추가
            for kw_entity in keyword_entities:
                is_overlapping = any(
                    (kw_entity.start >= e.start and kw_entity.start < e.end) or
                    (kw_entity.end > e.start and kw_entity.end <= e.end)
                    for e in result.health_entities
                )
                if not is_overlapping:
                    result.health_entities.append(kw_entity)
        else:
            # 키워드 매칭만 사용
            result.health_entities = self._find_health_keywords_direct(text)
        
        # 위치순 정렬
        result.health_entities.sort(key=lambda x: x.start)
        
        return result
    
    def get_health_terms(self, text: str, use_model: bool = True) -> List[Tuple[str, int, int]]:
        """건강 관련 용어만 추출 (간단한 인터페이스)
        
        Returns:
            List[(용어, 시작위치, 끝위치)]
        """
        result = self.process(text, use_model=use_model)
        return [(e.text, e.start, e.end) for e in result.health_entities]


# 싱글톤 인스턴스
_ner_processor: Optional[KoreanNERProcessor] = None


def get_ner_processor(
    model_name: str = "Leo97/KoELECTRA-small-v3-modu-ner",
    **kwargs
) -> KoreanNERProcessor:
    """NER 프로세서 싱글톤 인스턴스 반환"""
    global _ner_processor
    if _ner_processor is None:
        _ner_processor = KoreanNERProcessor(model_name=model_name, **kwargs)
    return _ner_processor
