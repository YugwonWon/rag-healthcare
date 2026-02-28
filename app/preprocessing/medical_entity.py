"""
바른 형태소 분석기 기반 의료 개체 인식 모듈
(Morphological Analysis-based Medical Entity Recognition)

기존 KoELECTRA NER을 대체하여:
1. bareunpy 형태소 분석 → 어근 복원 + 불용어 제거
2. 의료 용어 사전 매칭 (어근 기준, 카테고리 분류)
3. 복합 패턴 감지 (다중 형태소 조합)
4. LLM Fallback (복합 패턴에서 놓친 건강 관련 표현)
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

# ── bareunpy 설정 ──
BAREUN_API_KEY = "koba-6BVW24Q-2MIEHVQ-WQAVXJI-E2TRS7A"
BAREUN_HOST = "api.bareun.ai"
BAREUN_PORT = 443


class MedicalCategory(str, Enum):
    """의료 용어 카테고리"""
    SYMPTOM = "symptom"               # 증상 (통증, 어지러움 등)
    CONDITION = "condition"           # 질환/상태 (당뇨, 고혈압 등)
    BODY_PART = "body_part"           # 신체 부위 (무릎, 허리 등)
    MEDICATION = "medication"         # 약물 (혈압약, 수면제 등)
    TREATMENT = "treatment"           # 치료/검사 (수술, 진료 등)
    VITAL_SIGN = "vital_sign"         # 생체 신호 (혈압, 혈당 등)
    SLEEP = "sleep"                   # 수면 관련 (불면, 잠 등)
    NUTRITION = "nutrition"           # 영양/식이 (식욕, 체중 등)
    EMOTION = "emotion"               # 정서 (우울, 불안 등)
    COGNITIVE = "cognitive"           # 인지 (기억, 건망증 등)
    HYGIENE = "hygiene"               # 위생/일상 (목욕, 양치 등)
    ACTIVITY = "activity"             # 활동 (산책, 운동 등)
    EMERGENCY = "emergency"           # 응급 (의식, 호흡 등)


@dataclass
class MedicalEntity:
    """인식된 의료 개체"""
    text: str                         # 원문 텍스트
    lemma: str                        # 어근 (형태소 분석 결과)
    category: MedicalCategory         # 의료 카테고리
    pos_tag: str                      # 품사 태그 (NNG, VV 등)
    start: int                        # 원문 시작 위치
    end: int                          # 원문 끝 위치
    confidence: float = 1.0           # 신뢰도
    source: str = "dictionary"        # 인식 출처 (dictionary / pattern / llm)


@dataclass
class MorphAnalysisResult:
    """형태소 분석 + 의료 개체 인식 결과"""
    original_text: str
    morphemes: List[Tuple[str, str]]         # [(형태소, 태그), ...]
    nouns: List[str]                         # 추출된 명사
    verb_stems: List[str]                    # 동사/형용사 어근
    medical_entities: List[MedicalEntity] = field(default_factory=list)
    health_terms: List[str] = field(default_factory=list)  # 하위 호환용


# ═══════════════════════════════════════════════════════
#  의료 용어 사전 (어근 기반)
# ═══════════════════════════════════════════════════════

# 명사(NNG/NNP) 기반 사전: {어근: (카테고리, 설명)}
MEDICAL_NOUN_DICT: Dict[str, Tuple[MedicalCategory, str]] = {
    # ── 증상 ──
    "통증": (MedicalCategory.SYMPTOM, "통증"),
    "두통": (MedicalCategory.SYMPTOM, "두통"),
    "복통": (MedicalCategory.SYMPTOM, "복통"),
    "관절통": (MedicalCategory.SYMPTOM, "관절통"),
    "신경통": (MedicalCategory.SYMPTOM, "신경통"),
    "요통": (MedicalCategory.SYMPTOM, "요통"),
    "기침": (MedicalCategory.SYMPTOM, "기침"),
    "콧물": (MedicalCategory.SYMPTOM, "콧물"),
    "가래": (MedicalCategory.SYMPTOM, "가래"),
    "구토": (MedicalCategory.SYMPTOM, "구토"),
    "구역질": (MedicalCategory.SYMPTOM, "구역질"),
    "설사": (MedicalCategory.SYMPTOM, "설사"),
    "변비": (MedicalCategory.SYMPTOM, "변비"),
    "부종": (MedicalCategory.SYMPTOM, "부종"),
    "발열": (MedicalCategory.SYMPTOM, "발열"),
    "열": (MedicalCategory.SYMPTOM, "발열"),
    "경련": (MedicalCategory.SYMPTOM, "경련"),
    "발작": (MedicalCategory.SYMPTOM, "발작"),
    "출혈": (MedicalCategory.SYMPTOM, "출혈"),
    "멍": (MedicalCategory.SYMPTOM, "타박상"),
    "재채기": (MedicalCategory.SYMPTOM, "재채기"),
    "현기증": (MedicalCategory.SYMPTOM, "현기증"),
    "이명": (MedicalCategory.SYMPTOM, "이명"),
    "마비": (MedicalCategory.SYMPTOM, "마비"),
    "욕창": (MedicalCategory.SYMPTOM, "욕창"),
    "탈수": (MedicalCategory.SYMPTOM, "탈수"),
    "감기": (MedicalCategory.SYMPTOM, "감기"),
    "몸살": (MedicalCategory.SYMPTOM, "몸살"),
    "졸음": (MedicalCategory.SYMPTOM, "졸음"),
    "피로": (MedicalCategory.SYMPTOM, "피로"),
    "피곤": (MedicalCategory.SYMPTOM, "피로"),

    # ── 질환/상태 ──
    "당뇨": (MedicalCategory.CONDITION, "당뇨병"),
    "고혈압": (MedicalCategory.CONDITION, "고혈압"),
    "저혈압": (MedicalCategory.CONDITION, "저혈압"),
    "관절염": (MedicalCategory.CONDITION, "관절염"),
    "골다공증": (MedicalCategory.CONDITION, "골다공증"),
    "백내장": (MedicalCategory.CONDITION, "백내장"),
    "녹내장": (MedicalCategory.CONDITION, "녹내장"),
    "치매": (MedicalCategory.CONDITION, "치매"),
    "건망증": (MedicalCategory.CONDITION, "건망증"),
    "불면증": (MedicalCategory.CONDITION, "불면증"),
    "불면": (MedicalCategory.CONDITION, "불면증"),
    "난청": (MedicalCategory.CONDITION, "난청"),
    "요실금": (MedicalCategory.CONDITION, "요실금"),
    "변실금": (MedicalCategory.CONDITION, "변실금"),
    "폐렴": (MedicalCategory.CONDITION, "폐렴"),
    "천식": (MedicalCategory.CONDITION, "천식"),
    "암": (MedicalCategory.CONDITION, "암"),
    "우울증": (MedicalCategory.CONDITION, "우울증"),
    "골절": (MedicalCategory.CONDITION, "골절"),
    "낙상": (MedicalCategory.CONDITION, "낙상"),
    "탈모": (MedicalCategory.CONDITION, "탈모"),
    "갱년기": (MedicalCategory.CONDITION, "갱년기"),
    "노안": (MedicalCategory.CONDITION, "노안"),
    "비만": (MedicalCategory.CONDITION, "비만"),
    "수면무호흡": (MedicalCategory.CONDITION, "수면무호흡증"),
    "콜레스테롤": (MedicalCategory.CONDITION, "고콜레스테롤혈증"),

    # ── 신체 부위 ──
    "무릎": (MedicalCategory.BODY_PART, "무릎"),
    "허리": (MedicalCategory.BODY_PART, "허리"),
    "어깨": (MedicalCategory.BODY_PART, "어깨"),
    "목": (MedicalCategory.BODY_PART, "목"),
    "머리": (MedicalCategory.BODY_PART, "머리"),
    "손목": (MedicalCategory.BODY_PART, "손목"),
    "발목": (MedicalCategory.BODY_PART, "발목"),
    "관절": (MedicalCategory.BODY_PART, "관절"),
    "뼈": (MedicalCategory.BODY_PART, "뼈"),
    "근육": (MedicalCategory.BODY_PART, "근육"),
    "발톱": (MedicalCategory.BODY_PART, "발톱"),
    "손톱": (MedicalCategory.BODY_PART, "손톱"),
    "피부": (MedicalCategory.BODY_PART, "피부"),
    "잇몸": (MedicalCategory.BODY_PART, "잇몸"),
    "치아": (MedicalCategory.BODY_PART, "치아"),
    "눈": (MedicalCategory.BODY_PART, "눈"),
    "귀": (MedicalCategory.BODY_PART, "귀"),
    "폐": (MedicalCategory.BODY_PART, "폐"),
    "심장": (MedicalCategory.BODY_PART, "심장"),
    "간": (MedicalCategory.BODY_PART, "간"),
    "신장": (MedicalCategory.BODY_PART, "신장"),
    "위": (MedicalCategory.BODY_PART, "위"),
    "장": (MedicalCategory.BODY_PART, "장"),
    "다리": (MedicalCategory.BODY_PART, "다리"),
    "팔": (MedicalCategory.BODY_PART, "팔"),
    "가슴": (MedicalCategory.BODY_PART, "가슴"),
    "배": (MedicalCategory.BODY_PART, "배"),

    # ── 약물 ──
    "약": (MedicalCategory.MEDICATION, "약"),
    "수면제": (MedicalCategory.MEDICATION, "수면제"),
    "진통제": (MedicalCategory.MEDICATION, "진통제"),
    "항생제": (MedicalCategory.MEDICATION, "항생제"),
    "연고": (MedicalCategory.MEDICATION, "연고"),
    "주사": (MedicalCategory.MEDICATION, "주사"),
    "알약": (MedicalCategory.MEDICATION, "알약"),
    "물약": (MedicalCategory.MEDICATION, "물약"),
    "처방전": (MedicalCategory.MEDICATION, "처방전"),

    # ── 치료/검사 ──
    "수술": (MedicalCategory.TREATMENT, "수술"),
    "진료": (MedicalCategory.TREATMENT, "진료"),
    "검사": (MedicalCategory.TREATMENT, "검사"),
    "진단": (MedicalCategory.TREATMENT, "진단"),
    "치료": (MedicalCategory.TREATMENT, "치료"),
    "처방": (MedicalCategory.TREATMENT, "처방"),
    "입원": (MedicalCategory.TREATMENT, "입원"),
    "퇴원": (MedicalCategory.TREATMENT, "퇴원"),
    "재활": (MedicalCategory.TREATMENT, "재활"),

    # ── 생체 신호 ──
    "혈압": (MedicalCategory.VITAL_SIGN, "혈압"),
    "혈당": (MedicalCategory.VITAL_SIGN, "혈당"),
    "맥박": (MedicalCategory.VITAL_SIGN, "맥박"),
    "체온": (MedicalCategory.VITAL_SIGN, "체온"),
    "체중": (MedicalCategory.VITAL_SIGN, "체중"),

    # ── 수면 ──
    "잠": (MedicalCategory.SLEEP, "수면"),
    "수면": (MedicalCategory.SLEEP, "수면"),
    "낮잠": (MedicalCategory.SLEEP, "낮잠"),
    "악몽": (MedicalCategory.SLEEP, "악몽"),

    # ── 영양/식이 ──
    "식욕": (MedicalCategory.NUTRITION, "식욕"),
    "입맛": (MedicalCategory.NUTRITION, "식욕"),
    "식사": (MedicalCategory.NUTRITION, "식사"),
    "밥": (MedicalCategory.NUTRITION, "식사"),
    "영양": (MedicalCategory.NUTRITION, "영양"),
    "비타민": (MedicalCategory.NUTRITION, "비타민"),
    "소화": (MedicalCategory.NUTRITION, "소화"),

    # ── 정서 ──
    "우울": (MedicalCategory.EMOTION, "우울"),
    "불안": (MedicalCategory.EMOTION, "불안"),
    "걱정": (MedicalCategory.EMOTION, "걱정"),
    "스트레스": (MedicalCategory.EMOTION, "스트레스"),
    "눈물": (MedicalCategory.EMOTION, "눈물"),

    # ── 인지 ──
    "기억": (MedicalCategory.COGNITIVE, "기억"),
    "집중": (MedicalCategory.COGNITIVE, "집중"),
    "인지": (MedicalCategory.COGNITIVE, "인지"),
    "판단": (MedicalCategory.COGNITIVE, "판단"),

    # ── 위생/일상 ──
    "목욕": (MedicalCategory.HYGIENE, "목욕"),
    "샤워": (MedicalCategory.HYGIENE, "샤워"),
    "세수": (MedicalCategory.HYGIENE, "세수"),
    "양치": (MedicalCategory.HYGIENE, "양치"),
    "기저귀": (MedicalCategory.HYGIENE, "기저귀"),
    "화장실": (MedicalCategory.HYGIENE, "화장실"),
    "소변": (MedicalCategory.HYGIENE, "소변"),
    "대변": (MedicalCategory.HYGIENE, "대변"),

    # ── 활동 ──
    "산책": (MedicalCategory.ACTIVITY, "산책"),
    "운동": (MedicalCategory.ACTIVITY, "운동"),
    "걷기": (MedicalCategory.ACTIVITY, "걷기"),
    "체조": (MedicalCategory.ACTIVITY, "체조"),
    "스트레칭": (MedicalCategory.ACTIVITY, "스트레칭"),

    # ── 응급 ──
    "응급": (MedicalCategory.EMERGENCY, "응급"),
    "구급": (MedicalCategory.EMERGENCY, "구급"),
    "의식": (MedicalCategory.EMERGENCY, "의식"),
    "호흡": (MedicalCategory.EMERGENCY, "호흡"),

    # ── 의료 시설 ──
    "병원": (MedicalCategory.TREATMENT, "병원"),
    "의원": (MedicalCategory.TREATMENT, "의원"),
    "한의원": (MedicalCategory.TREATMENT, "한의원"),
    "응급실": (MedicalCategory.TREATMENT, "응급실"),
    "약국": (MedicalCategory.TREATMENT, "약국"),
}

# 동사/형용사(VV/VA) 어근 사전: {어근: (카테고리, 설명)}
MEDICAL_VERB_DICT: Dict[str, Tuple[MedicalCategory, str]] = {
    # ── 증상 동사 ──
    "아프": (MedicalCategory.SYMPTOM, "통증"),
    "쑤시": (MedicalCategory.SYMPTOM, "쑤심"),
    "결리": (MedicalCategory.SYMPTOM, "결림"),
    "저리": (MedicalCategory.SYMPTOM, "저림"),
    "가렵": (MedicalCategory.SYMPTOM, "가려움"),
    "붓": (MedicalCategory.SYMPTOM, "부종"),
    "어지럽": (MedicalCategory.SYMPTOM, "어지러움"),
    "어지러": (MedicalCategory.SYMPTOM, "어지러움"),
    "메스껍": (MedicalCategory.SYMPTOM, "메스꺼움"),
    "토하": (MedicalCategory.SYMPTOM, "구토"),
    "기침하": (MedicalCategory.SYMPTOM, "기침"),
    "떨리": (MedicalCategory.SYMPTOM, "떨림"),
    "욱신거리": (MedicalCategory.SYMPTOM, "욱신거림"),
    "화끈거리": (MedicalCategory.SYMPTOM, "화끈거림"),

    # ── 낙상/운동 관련 ──
    "넘어지": (MedicalCategory.CONDITION, "낙상"),
    "미끄러지": (MedicalCategory.CONDITION, "낙상"),
    "쓰러지": (MedicalCategory.CONDITION, "낙상"),
    "부러지": (MedicalCategory.CONDITION, "골절"),
    "비틀거리": (MedicalCategory.CONDITION, "균형 장애"),
    "다치": (MedicalCategory.CONDITION, "부상"),

    # ── 수면 ──
    "잠들": (MedicalCategory.SLEEP, "수면 장애"),

    # ── 인지 ──
    "잊어버리": (MedicalCategory.COGNITIVE, "기억력 저하"),
    "깜빡하": (MedicalCategory.COGNITIVE, "기억력 저하"),
    "헷갈리": (MedicalCategory.COGNITIVE, "인지 혼란"),

    # ── 정서 ──
    "슬프": (MedicalCategory.EMOTION, "슬픔"),
    "외롭": (MedicalCategory.EMOTION, "외로움"),
    "무섭": (MedicalCategory.EMOTION, "두려움"),
    "두렵": (MedicalCategory.EMOTION, "두려움"),
    "짜증나": (MedicalCategory.EMOTION, "짜증"),
    "화나": (MedicalCategory.EMOTION, "분노"),

    # ── 영양 ──
    "먹": (MedicalCategory.NUTRITION, "식사"),
    "삼키": (MedicalCategory.NUTRITION, "연하"),

    # ── 약물 ──
    "복용하": (MedicalCategory.MEDICATION, "복용"),
}

# ═══════════════════════════════════════════════════════
#  복합 패턴 (다중 형태소 조합)
# ═══════════════════════════════════════════════════════

# (정규식 패턴, 카테고리, 설명)
# 형태소 분석 후 "어근/태그 어근/태그" 형식으로 매칭
COMPOUND_PATTERNS: List[Tuple[str, MedicalCategory, str]] = [
    # 수면 복합 패턴
    (r"잠.{0,4}못.{0,4}(자|들|잤)", MedicalCategory.SLEEP, "수면 장애"),
    (r"잠.{0,2}(안|못)\s*(오|와|들)", MedicalCategory.SLEEP, "수면 장애"),
    (r"밤.{0,4}(깨|뒤척|잠)", MedicalCategory.SLEEP, "수면 장애"),
    (r"새벽.{0,4}(깨|눈)", MedicalCategory.SLEEP, "수면 장애"),

    # 호흡 복합 패턴
    (r"숨.{0,4}(못|안|차|찬|쉬)", MedicalCategory.EMERGENCY, "호흡 곤란"),
    (r"숨.{0,2}(가쁘|막히)", MedicalCategory.EMERGENCY, "호흡 곤란"),

    # 식사 복합 패턴
    (r"밥.{0,4}(못|안|잘).{0,4}(먹|넘)", MedicalCategory.NUTRITION, "식사 곤란"),
    (r"입맛.{0,4}(없|안)", MedicalCategory.NUTRITION, "식욕 부진"),
    (r"소화.{0,4}(안|못|잘)", MedicalCategory.NUTRITION, "소화 불량"),

    # 배변 복합 패턴
    (r"변.{0,4}(안|못).{0,4}(나|보)", MedicalCategory.SYMPTOM, "배변 곤란"),
    (r"소변.{0,4}(못\s*참|자주|실수)", MedicalCategory.CONDITION, "요실금"),

    # 감각 복합 패턴
    (r"(눈|시력).{0,4}(침침|흐릿|안\s*보)", MedicalCategory.CONDITION, "시력 저하"),
    (r"귀.{0,4}(안\s*들|잘\s*안)", MedicalCategory.CONDITION, "청력 저하"),
    (r"(손|발).{0,4}(저리|감각|떨리)", MedicalCategory.SYMPTOM, "말초 신경 증상"),

    # 통증 복합 패턴  
    (r"(머리|가슴|배).{0,4}(아프|아파|통증)", MedicalCategory.SYMPTOM, "통증"),

    # 약 복합 패턴
    (r"약.{0,4}(너무\s*많|과다|과량)", MedicalCategory.EMERGENCY, "약물 과다"),
    (r"약.{0,4}(먹|복용|드시).{0,4}(안|않|못)", MedicalCategory.MEDICATION, "복약 누락"),
    (r"부작용.{0,4}(있|나타|생기|심하)", MedicalCategory.MEDICATION, "약물 부작용"),

    # 발톱/피부 복합 패턴
    (r"발톱.{0,6}(안|안쪽|파고|들어)", MedicalCategory.CONDITION, "내향성 발톱"),
    (r"발톱.{0,6}(두꺼|변형|휘)", MedicalCategory.CONDITION, "발톱 변형"),
    (r"피부.{0,6}(가려|건조|트러블)", MedicalCategory.SYMPTOM, "피부 증상"),
    (r"머리카락.{0,4}(빠지|빠져|탈모)", MedicalCategory.CONDITION, "탈모"),

    # 기력 복합 패턴
    (r"기운.{0,4}(없|빠지|저하)", MedicalCategory.SYMPTOM, "기력 저하"),
    (r"(기침|콧물).{0,4}(오래|안\s*멈|계속)", MedicalCategory.SYMPTOM, "지속 증상"),

    # 혈당/혈압 복합 패턴
    (r"혈당.{0,4}(높|올라|떨어|낮)", MedicalCategory.VITAL_SIGN, "혈당 이상"),
    (r"혈압.{0,4}(높|올라|떨어|낮)", MedicalCategory.VITAL_SIGN, "혈압 이상"),

    # 인지 복합 패턴
    (r"(길|방향).{0,4}(잃|모르|헤매)", MedicalCategory.COGNITIVE, "방향 감각 상실"),
    (r"(사람|이름).{0,4}(기억|모르|잊)", MedicalCategory.COGNITIVE, "인지 기능 저하"),
]


# ═══════════════════════════════════════════════════════
#  불용어 (형태소 분석 후 제거)
# ═══════════════════════════════════════════════════════

STOPWORDS = {
    # 조사
    "이", "가", "을", "를", "에", "의", "도", "만", "은", "는",
    "에서", "으로", "와", "과", "로", "서", "고", "게", "요",
    # 일반 부사
    "좀", "잘", "많이", "너무", "정말", "진짜", "그냥", "되게",
    "아주", "매우", "약간", "조금",
    # 대명사/지시사
    "저", "나", "제", "내", "우리", "거", "것", "데",
    # 일반 동사 어근 (의료 무관)
    "하", "되", "있", "없", "이", "보", "알", "가",
    "오", "주", "받", "내", "나오",
}

# 의료 맥락에서만 의미 있는 동사 (불용어에서 제외)
# "먹" — NUTRITION 사전에 이미 등록, "자" — SLEEP에서 복합패턴으로 처리
MEDICAL_CONTEXT_VERBS = {"먹", "자", "삼키", "토하"}


# ═══════════════════════════════════════════════════════
#  바른 형태소 분석기 래퍼
# ═══════════════════════════════════════════════════════

class BareunMorphAnalyzer:
    """bareunpy 기반 형태소 분석기
    
    - 어근 복원 (활용형 → 기본형)
    - 불용어 제거
    - 의료 용어 사전 매칭 (명사/동사 어근)
    - 복합 패턴 감지
    """

    def __init__(self):
        self._tagger = None
        self._initialized = False

    def _ensure_tagger(self):
        """Tagger 지연 초기화"""
        if not self._initialized:
            try:
                from bareunpy import Tagger
                self._tagger = Tagger(BAREUN_API_KEY, BAREUN_HOST, port=BAREUN_PORT)
                self._initialized = True
                logger.info("✅ 바른 형태소 분석기 초기화 완료")
            except Exception as e:
                logger.error(f"바른 형태소 분석기 초기화 실패: {e}")
                self._tagger = None

    def analyze(self, text: str) -> MorphAnalysisResult:
        """텍스트 형태소 분석 + 의료 개체 인식
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            MorphAnalysisResult: 분석 결과
        """
        result = MorphAnalysisResult(
            original_text=text,
            morphemes=[],
            nouns=[],
            verb_stems=[],
        )

        self._ensure_tagger()

        if self._tagger is None:
            # bareunpy 실패 시 폴백: 키워드 직접 매칭
            logger.warning("바른 형태소 분석기 사용 불가 → 키워드 직접 매칭 폴백")
            result.medical_entities = self._fallback_keyword_match(text)
            result.health_terms = [e.text for e in result.medical_entities]
            return result

        try:
            # 1) 형태소 분석
            pos_result = self._tagger.pos(text)
            result.morphemes = pos_result

            # 2) 명사 추출
            result.nouns = [morph for morph, tag in pos_result if tag.startswith("NN")]

            # 3) 동사/형용사 어근 추출
            result.verb_stems = [
                morph for morph, tag in pos_result
                if tag in ("VV", "VA", "XSV", "XSA")
            ]

            # 4) 의료 명사 사전 매칭
            entities = self._match_noun_dict(text, pos_result)

            # 5) 의료 동사 사전 매칭
            entities.extend(self._match_verb_dict(text, pos_result))

            # 6) 복합 패턴 매칭
            entities.extend(self._match_compound_patterns(text))

            # 7) 중복 제거 (같은 위치 범위의 개체 중 더 구체적인 것만 유지)
            result.medical_entities = self._deduplicate_entities(entities)

            # 하위 호환: health_terms 리스트
            result.health_terms = list(set(
                e.lemma for e in result.medical_entities
            ))

        except Exception as e:
            logger.error(f"형태소 분석 오류: {e}")
            result.medical_entities = self._fallback_keyword_match(text)
            result.health_terms = [e.text for e in result.medical_entities]

        return result

    def _match_noun_dict(
        self, text: str, pos_result: List[Tuple[str, str]]
    ) -> List[MedicalEntity]:
        """명사 형태소 → 의료 명사 사전 매칭"""
        entities = []
        for morph, tag in pos_result:
            if not tag.startswith("NN"):
                continue
            if morph in MEDICAL_NOUN_DICT:
                category, desc = MEDICAL_NOUN_DICT[morph]
                # 원문에서 위치 찾기
                start = text.find(morph)
                if start == -1:
                    start = 0
                entities.append(MedicalEntity(
                    text=morph,
                    lemma=morph,
                    category=category,
                    pos_tag=tag,
                    start=start,
                    end=start + len(morph),
                    confidence=1.0,
                    source="dictionary",
                ))
        return entities

    def _match_verb_dict(
        self, text: str, pos_result: List[Tuple[str, str]]
    ) -> List[MedicalEntity]:
        """동사/형용사 어근 → 의료 동사 사전 매칭"""
        entities = []
        for morph, tag in pos_result:
            if tag not in ("VV", "VA", "XSV", "XSA"):
                continue
            if morph in MEDICAL_VERB_DICT:
                category, desc = MEDICAL_VERB_DICT[morph]
                # 불용어에 해당하면서 의료 맥락 동사가 아니면 스킵
                if morph in STOPWORDS and morph not in MEDICAL_CONTEXT_VERBS:
                    continue
                start = text.find(morph)
                if start == -1:
                    start = 0
                entities.append(MedicalEntity(
                    text=morph,
                    lemma=desc,
                    category=category,
                    pos_tag=tag,
                    start=start,
                    end=start + len(morph),
                    confidence=0.9,
                    source="dictionary",
                ))
        return entities

    def _match_compound_patterns(self, text: str) -> List[MedicalEntity]:
        """원문에서 복합 패턴 매칭"""
        entities = []
        for pattern, category, desc in COMPOUND_PATTERNS:
            match = re.search(pattern, text)
            if match:
                entities.append(MedicalEntity(
                    text=match.group(0),
                    lemma=desc,
                    category=category,
                    pos_tag="COMPOUND",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                    source="pattern",
                ))
        return entities

    def _deduplicate_entities(
        self, entities: List[MedicalEntity]
    ) -> List[MedicalEntity]:
        """겹치는 개체 중복 제거
        
        같은 위치 범위: pattern > dictionary
        같은 lemma: 첫 번째만 유지
        """
        if not entities:
            return []

        # source 우선순위: pattern > dictionary > llm
        priority = {"pattern": 3, "dictionary": 2, "llm": 1}

        # 위치순 정렬 후 겹침 제거
        sorted_ents = sorted(entities, key=lambda e: (e.start, -priority.get(e.source, 0)))
        result = []
        seen_lemmas = set()

        for ent in sorted_ents:
            # 동일 lemma 중복 제거
            if ent.lemma in seen_lemmas:
                continue
            # 이전 개체와 위치 겹침 확인
            if result and ent.start < result[-1].end:
                # 우선순위 높으면 교체
                if priority.get(ent.source, 0) > priority.get(result[-1].source, 0):
                    seen_lemmas.discard(result[-1].lemma)
                    result[-1] = ent
                    seen_lemmas.add(ent.lemma)
                continue
            result.append(ent)
            seen_lemmas.add(ent.lemma)

        return result

    def _fallback_keyword_match(self, text: str) -> List[MedicalEntity]:
        """bareunpy 실패 시 폴백: 키워드 직접 매칭"""
        entities = []
        text_lower = text.lower()

        for keyword, (category, desc) in MEDICAL_NOUN_DICT.items():
            if keyword in text_lower:
                start = text_lower.find(keyword)
                entities.append(MedicalEntity(
                    text=keyword,
                    lemma=keyword,
                    category=category,
                    pos_tag="NNG",
                    start=start,
                    end=start + len(keyword),
                    confidence=0.8,
                    source="fallback",
                ))

        # 복합 패턴도 폴백으로 실행
        entities.extend(self._match_compound_patterns(text))
        return self._deduplicate_entities(entities)

    def get_health_terms(self, text: str) -> List[Tuple[str, int, int]]:
        """건강 관련 용어 추출 (하위 호환 인터페이스)
        
        기존 KoreanNERProcessor.get_health_terms()와 동일한 시그니처
        
        Returns:
            List[(용어, 시작위치, 끝위치)]
        """
        result = self.analyze(text)
        return [(e.text, e.start, e.end) for e in result.medical_entities]


# ── 싱글톤 ──
_analyzer: Optional[BareunMorphAnalyzer] = None


def get_morph_analyzer() -> BareunMorphAnalyzer:
    """형태소 분석기 싱글톤 반환"""
    global _analyzer
    if _analyzer is None:
        _analyzer = BareunMorphAnalyzer()
    return _analyzer
