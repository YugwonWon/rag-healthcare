"""
PII(개인식별정보) 감지 및 삭제 모듈
(Personal Identifiable Information Detection & Redaction)

HIPAA Safe Harbor 방법을 한국어 돌봄 맥락에 적용:
1. 규칙 기반 키워드 매칭 (가족 구성원, 낙인 건강 상태, 민감 주제)
2. 정규식 기반 PII 패턴 감지 (이름, 전화번호, 주소, 주민번호)
3. 형태소 분석 기반 민감 용어 어근 감지

마스킹 전략:
- 가명처리(Pseudonymization): 이름 → [사용자A], 가족 → [가족1]
- 일반화(Generalization): 나이 → 연대, 주소 → [지역]
- 억제(Suppression): 주민번호, 전화번호 → 완전 삭제
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from app.logger import get_logger

logger = get_logger(__name__)


class PIIType(str, Enum):
    """PII 유형"""
    NAME = "name"                     # 한국어 이름
    PHONE = "phone"                   # 전화번호
    ADDRESS = "address"               # 주소
    SSN = "ssn"                       # 주민등록번호
    AGE_SPECIFIC = "age_specific"     # 구체적 나이
    FAMILY_REFERENCE = "family_ref"   # 가족 구성원 언급
    STIGMATIZED_CONDITION = "stigma"  # 낙인 건강 상태
    EMOTIONAL_SENSITIVE = "emotional" # 정서적 민감 주제
    HOSPITAL_NAME = "hospital"        # 병원명
    DATE_SPECIFIC = "date_specific"   # 구체적 날짜


class MaskingStrategy(str, Enum):
    """마스킹 전략"""
    PSEUDONYMIZE = "pseudonymize"     # 가명처리
    GENERALIZE = "generalize"         # 일반화
    SUPPRESS = "suppress"             # 완전 삭제


@dataclass
class PIIDetection:
    """감지된 PII 항목"""
    pii_type: PIIType
    original_text: str
    start: int
    end: int
    masking_strategy: MaskingStrategy
    masked_text: str
    confidence: float = 1.0


@dataclass
class RedactionResult:
    """삭제 처리 결과"""
    original_text: str
    redacted_text: str
    detections: List[PIIDetection] = field(default_factory=list)
    pii_count: int = 0
    clinical_terms_preserved: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════
#  한국어 성씨 목록 (상위 빈도)
# ═══════════════════════════════════════════════════════
KOREAN_SURNAMES = {
    "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
    "한", "오", "서", "신", "권", "황", "안", "송", "류", "전",
    "홍", "고", "문", "양", "손", "배", "백", "허", "유", "남",
    "심", "노", "하", "곽", "성", "차", "주", "우", "구", "민",
    "진", "나", "변", "원", "천", "방", "공", "현", "함", "석",
}

# ═══════════════════════════════════════════════════════
#  가족 구성원 키워드
# ═══════════════════════════════════════════════════════
FAMILY_KEYWORDS = {
    "아들": "[가족1]",
    "딸": "[가족2]",
    "며느리": "[가족3]",
    "사위": "[가족4]",
    "손자": "[가족5]",
    "손녀": "[가족6]",
    "아내": "[가족7]",
    "남편": "[가족8]",
    "할머니": "[가족9]",
    "할아버지": "[가족10]",
    "어머니": "[가족11]",
    "아버지": "[가족12]",
    "엄마": "[가족11]",
    "아빠": "[가족12]",
    "형": "[가족13]",
    "누나": "[가족14]",
    "동생": "[가족15]",
    "오빠": "[가족16]",
    "언니": "[가족17]",
    "시어머니": "[가족18]",
    "시아버지": "[가족19]",
    "큰아들": "[가족1]",
    "작은아들": "[가족1]",
    "큰딸": "[가족2]",
    "작은딸": "[가족2]",
}

# ═══════════════════════════════════════════════════════
#  낙인 건강 상태 용어 (직접 언급 → 완곡 표현으로)
# ═══════════════════════════════════════════════════════
STIGMATIZED_CONDITIONS = {
    "치매": "[인지기능 관련]",
    "알츠하이머": "[인지기능 관련]",
    "요실금": "[배뇨기능 관련]",
    "변실금": "[배변기능 관련]",
    "우울증": "[정서건강 관련]",
    "조현병": "[정신건강 관련]",
    "자살": "[정신건강 위험]",
    "성병": "[감염 관련]",
    "에이즈": "[감염 관련]",
    "간염": "[감염 관련]",
    "결핵": "[호흡기 감염]",
    "암": "[중대질환 관련]",
    "말기": "[중대질환 관련]",
}

# ═══════════════════════════════════════════════════════
#  정서적 민감 표현 — VA 어근 (bareunpy 형태소 분석 기반)
# ═══════════════════════════════════════════════════════
EMOTIONAL_VA_STEMS = {
    # ㅂ불규칙 형용사 (원형만 등록, 활용형은 자동 생성)
    "부끄럽": "[정서표현]",
    "두렵": "[정서표현]",
    "무섭": "[정서표현]",
    "외롭": "[정서표현]",
    "서럽": "[정서표현]",
    "슬프": "[정서표현]",
    "수치스럽": "[정서표현]",
    "당혹스럽": "[정서표현]",
    "걱정스럽": "[정서표현]",
    "아프": "[정서표현]",
    "울적하": "[정서표현]",
    "안타깝": "[정서표현]",
    "괴롭": "[정서표현]",
    "고통스럽": "[정서표현]",
    "쑥스럽": "[정서표현]",
    "경악스럽": "[정서표현]",
    "절망스럽": "[정서표현]",
    "서글프": "[정서표현]",
    "원망스럽": "[정서표현]",
    "한스럽": "[정서표현]",
    "한탄스럽": "[정서표현]",
    "어이없": "[정서표현]",
    # 하다형 형용사
    "상처받": "[정서표현]",
    "창피하": "[정서표현]",
    "민망하": "[정서표현]",
    "비참하": "[정서표현]",
    "답답하": "[정서표현]",
    "속상하": "[정서표현]",
    "불안하": "[정서표현]",
    "서운하": "[정서표현]",
    "억울하": "[정서표현]",
    "쓸쓸하": "[정서표현]",
    "허전하": "[정서표현]",
    "고뇌하": "[정서표현]",
    "섭섭하": "[정서표현]",
    "막막하": "[정서표현]",
    "우울하": "[정서표현]",
    "초조하": "[정서표현]",
    "조마조마하": "[정서표현]",
    "무력하": "[정서표현]",
    "공허하": "[정서표현]",
    "심란하": "[정서표현]",
    "참담하": "[정서표현]",
    "암담하": "[정서표현]",
    "원통하": "[정서표현]",
    "비통하": "[정서표현]",
    "처참하": "[정서표현]",
    "처량하": "[정서표현]",
    "후회하": "[정서표현]",
    "야속하": "[정서표현]",
    "참혹하": "[정서표현]",
    "무기력하": "[정서표현]",
    "얼떨떨하": "[정서표현]",
    "절박하": "[정서표현]",
    "혐오하": "[정서표현]",
    "애통하": "[정서표현]",
}

# ═══════════════════════════════════════════════════════
#  정서적 민감 표현 — 키워드 기반 (regex 매칭)
# ═══════════════════════════════════════════════════════
EMOTIONAL_KEYWORD_STEMS = {
    "자괴감": "[정서표현]",
    "절망": "[정서표현]",
    "치욕": "[정서표현]",
    "수치심": "[정서표현]",
    "당혹": "[정서표현]",
    "자존심": "[정서표현]",
    "화가 나": "[정서표현]",
    "울고 싶": "[정서표현]",
    "눈물이 나": "[정서표현]",
    "좌절감": "[정서표현]",
    "무기력": "[정서표현]",
    "트라우마": "[정서표현]",
    "공포": "[정서표현]",
    "고립감": "[정서표현]",
    "상실감": "[정서표현]",
    "열등감": "[정서표현]",
    "죄책감": "[정서표현]",
    "수모": "[정서표현]",
    "모멸": "[정서표현]",
    "모멸감": "[정서표현]",
    # bareunpy VV/NNG — VA 태그 미발생 항목 (키워드 regex로 검색)
    "짜증": "[정서표현]",
    "겁나": "[정서표현]",
    "화나": "[정서표현]",
    "낙담": "[정서표현]",
    "상심": "[정서표현]",
    "좌절": "[정서표현]",
    "지치": "[정서표현]",
    "가슴 아프": "[정서표현]",
    "가슴 아파": "[정서표현]",
    "눈물": "[정서표현]",
    "처지": "[정서표현]",
    "참을 수 없": "[정서표현]",
    "견디기 힘들": "[정서표현]",
    # 위기 표현
    "죽고 싶": "[위기표현]",
    "살기 싫": "[위기표현]",
    "자해": "[위기표현]",
    "목숨": "[위기표현]",
    "극단적": "[위기표현]",
}


def _build_stem_surface_regex(stem):
    """형용사 어근에서 표면형 매칭용 정규식 문자열 생성

    하다형(답답하→답답해), ㅂ불규칙(두렵→두려워),
    ㅡ불규칙(슬프→슬퍼)의 활용형을 자동으로 커버.
    """
    # 하다형: '답답하' → r'답답[가-힣]{0,4}'
    if stem.endswith("하"):
        return rf'{re.escape(stem[:-1])}[가-힣]{{0,4}}'

    last = stem[-1]
    code = ord(last) - 0xAC00
    variants = [re.escape(stem)]

    if 0 <= code < 11172:
        final = code % 28
        medial = (code // 28) % 21
        # ㅂ불규칙 (종성 ㅂ=17): 두렵→두려, 무섭→무서
        if final == 17:
            variants.append(re.escape(stem[:-1] + chr(ord(last) - 17)))
        # ㅡ불규칙 (중성 ㅡ=18, 종성 없음): 슬프→슬퍼
        elif final == 0 and medial == 18:
            variants.append(re.escape(stem[:-1] + chr(ord(last) - 14 * 28)))

    alt = "|".join(variants)
    return rf'(?:{alt})[가-힣]{{0,4}}'


# ㅂ불규칙 "~워하다" VV 파생형 → VA 어근 매핑 (자동 생성)
# 예: 부끄러워하(VV) → 부끄럽, 두려워하(VV) → 두렵
_BIEUP_VV_TO_VA = {}
for _stem in EMOTIONAL_VA_STEMS:
    _last = _stem[-1]
    _code = ord(_last) - 0xAC00
    if 0 <= _code < 11172 and _code % 28 == 17:  # ㅂ batchim
        _no_bieup = chr(ord(_last) - 17)
        _vv_form = _stem[:-1] + _no_bieup + "워하"
        _BIEUP_VV_TO_VA[_vv_form] = _stem


class PIIRedactor:
    """PII 감지 및 삭제 처리기
    
    논문 Dimension 3 - Conversation Data Sanitization:
    - 규칙 기반 키워드 매칭
    - 정규식 기반 PII 패턴 감지
    - 형태소 분석 기반 어근 감지 (선택적)
    """
    
    def __init__(self, use_morphology: bool = False, morph_analyzer=None):
        """
        Args:
            use_morphology: 형태소 분석 기반 어근 감지 사용 여부
            morph_analyzer: BareunMorphAnalyzer 인스턴스 (외부 주입)
        """
        self.use_morphology = use_morphology
        self.morph_analyzer = morph_analyzer
        self._family_counter = 0
        self._name_counter = 0
        self._name_map: Dict[str, str] = {}
        self._family_map: Dict[str, str] = {}
        
        # bareunpy Tagger (지연 초기화)
        self._tagger = None
        self._tagger_initialized = False
        
        # 정규식 패턴 사전 컴파일
        self._compile_patterns()
    
    def _ensure_tagger(self):
        """bareunpy Tagger 지연 초기화"""
        if not self._tagger_initialized:
            self._tagger_initialized = True
            try:
                from bareunpy import Tagger
                from app.preprocessing.medical_entity import BAREUN_API_KEY, BAREUN_HOST, BAREUN_PORT
                self._tagger = Tagger(BAREUN_API_KEY, BAREUN_HOST, port=BAREUN_PORT)
                logger.info("PII Redactor: 바른 형태소 분석기 초기화 완료")
            except Exception as e:
                logger.warning(f"PII Redactor: 바른 형태소 분석기 사용 불가 → 키워드 폴백 ({e})")
                self._tagger = None
    
    def _get_pos_result(self, text: str) -> Optional[list]:
        """bareunpy POS 분석 결과 반환 (전체 형태소 리스트)
        
        Returns:
            [(형태소, 품사태그), ...] 리스트, 또는 bareunpy 불가 시 None
        """
        self._ensure_tagger()
        if self._tagger is None:
            return None
        try:
            return self._tagger.pos(text)
        except Exception:
            return None
    
    def _compile_patterns(self):
        """정규식 패턴 컴파일"""
        # 한국어 이름 패턴: 성(1자) + 이름(1-2자) 한글 = 총 2-3자
        # 뒤에 조사/어미가 올 수 있으므로 lookahead로 조사 배제
        surname_pattern = "|".join(re.escape(s) for s in KOREAN_SURNAMES)
        self._name_pattern = re.compile(
            rf'(?<![가-힣])({surname_pattern})[가-힣]{{1,2}}(?=[^가-힣]|$)'
        )
        
        # 전화번호: 010-XXXX-XXXX 또는 02-XXX-XXXX 등
        self._phone_pattern = re.compile(
            r'(?:0\d{1,2})[-.\s]?(?:\d{3,4})[-.\s]?(?:\d{4})'
        )
        
        # 주민등록번호: XXXXXX-XXXXXXX
        self._ssn_pattern = re.compile(
            r'\d{6}[-\s]?[1-4]\d{6}'
        )
        
        # 주소 패턴: 시/도 + 구/군 + 동/읍/면 또는 ~로/~길
        # 최소 2단계 행정구역이 필요 (예: 서울시 강남구, 수원시 팔달구 인계동)
        self._address_pattern = re.compile(
            r'(?:[가-힣]{1,5}(?:특별시|광역시|도))\s*'
            r'(?:[가-힣]{1,5}(?:시|군|구))\s*'
            r'(?:[가-힣]{1,5}(?:동|읍|면|로|길))?'
            r'(?:\s*\d+(?:-\d+)?(?:번지)?)?'
            r'|'
            r'(?:[가-힣]{1,5}시)\s+'
            r'(?:[가-힣]{1,5}(?:구|군))\s*'
            r'(?:[가-힣]{1,5}(?:동|읍|면|로|길))'
            r'(?:\s*\d+(?:-\d+)?(?:번지)?)?'
        )
        
        # 구체적 나이: XX살, XX세
        self._age_pattern = re.compile(
            r'(\d{1,3})\s*(?:살|세)'
        )
        
        # 구체적 날짜: YYYY년 MM월, MM월 DD일
        self._date_pattern = re.compile(
            r'(?:\d{4}년\s*)?(?:\d{1,2}월)\s*(?:\d{1,2}일)?'
        )
        
        # 병원명: ~병원, ~의원, ~클리닉
        self._hospital_pattern = re.compile(
            r'[가-힣]{2,10}(?:대학)?(?:병원|의원|클리닉|센터|약국)'
        )
        
        # 정서적 민감 표현 — VA 어근별 표면형 regex
        self._va_stem_regexes = {}
        for stem in EMOTIONAL_VA_STEMS:
            self._va_stem_regexes[stem] = re.compile(
                _build_stem_surface_regex(stem)
            )
        # 정서적 민감 표현 — 키워드 regex
        self._kw_stem_regexes = {}
        for kw in EMOTIONAL_KEYWORD_STEMS:
            self._kw_stem_regexes[kw] = re.compile(
                rf'{re.escape(kw)}[가-힣]{{0,4}}'
            )
    
    def detect_pii(self, text: str) -> List[PIIDetection]:
        """텍스트에서 PII 감지
        
        Args:
            text: 입력 텍스트
        
        Returns:
            감지된 PII 목록 (위치순 정렬)
        """
        detections: List[PIIDetection] = []
        
        # 1. 주민등록번호 (최우선 - 가장 민감)
        for m in self._ssn_pattern.finditer(text):
            detections.append(PIIDetection(
                pii_type=PIIType.SSN,
                original_text=m.group(),
                start=m.start(), end=m.end(),
                masking_strategy=MaskingStrategy.SUPPRESS,
                masked_text="[삭제됨]",
                confidence=0.95,
            ))
        
        # 2. 전화번호
        for m in self._phone_pattern.finditer(text):
            if not self._overlaps(detections, m.start(), m.end()):
                detections.append(PIIDetection(
                    pii_type=PIIType.PHONE,
                    original_text=m.group(),
                    start=m.start(), end=m.end(),
                    masking_strategy=MaskingStrategy.SUPPRESS,
                    masked_text="[삭제됨]",
                    confidence=0.95,
                ))
        
        # 3. 주소
        for m in self._address_pattern.finditer(text):
            if not self._overlaps(detections, m.start(), m.end()):
                detections.append(PIIDetection(
                    pii_type=PIIType.ADDRESS,
                    original_text=m.group(),
                    start=m.start(), end=m.end(),
                    masking_strategy=MaskingStrategy.GENERALIZE,
                    masked_text="[지역]",
                    confidence=0.9,
                ))
        
        # bareunpy POS 분석 (1회 호출로 이름·정서 감지에 공유)
        pos_result = self._get_pos_result(text)
        nnp_set = {morph for morph, tag in pos_result if tag == "NNP"} if pos_result else None

        # 4. 한국어 이름
        
        for m in self._name_pattern.finditer(text):
            if not self._overlaps(detections, m.start(), m.end()):
                name = m.group()
                # NNP 기반 판별 (bareunpy 가용 시) 또는 키워드 폴백
                if self._is_name_by_pos(name, nnp_set, text, m.end()):
                    masked = self._get_name_pseudonym(name)
                    detections.append(PIIDetection(
                        pii_type=PIIType.NAME,
                        original_text=name,
                        start=m.start(), end=m.end(),
                        masking_strategy=MaskingStrategy.PSEUDONYMIZE,
                        masked_text=masked,
                        confidence=0.9 if nnp_set is not None else 0.7,
                    ))
        
        # 5. 구체적 나이 → 연대로 일반화
        for m in self._age_pattern.finditer(text):
            if not self._overlaps(detections, m.start(), m.end()):
                age = int(m.group(1))
                decade = (age // 10) * 10
                detections.append(PIIDetection(
                    pii_type=PIIType.AGE_SPECIFIC,
                    original_text=m.group(),
                    start=m.start(), end=m.end(),
                    masking_strategy=MaskingStrategy.GENERALIZE,
                    masked_text=f"{decade}대",
                    confidence=0.9,
                ))
        
        # 6. 구체적 날짜
        for m in self._date_pattern.finditer(text):
            if not self._overlaps(detections, m.start(), m.end()):
                detections.append(PIIDetection(
                    pii_type=PIIType.DATE_SPECIFIC,
                    original_text=m.group(),
                    start=m.start(), end=m.end(),
                    masking_strategy=MaskingStrategy.GENERALIZE,
                    masked_text="[날짜]",
                    confidence=0.85,
                ))
        
        # 7. 병원명
        for m in self._hospital_pattern.finditer(text):
            if not self._overlaps(detections, m.start(), m.end()):
                detections.append(PIIDetection(
                    pii_type=PIIType.HOSPITAL_NAME,
                    original_text=m.group(),
                    start=m.start(), end=m.end(),
                    masking_strategy=MaskingStrategy.GENERALIZE,
                    masked_text="[의료기관]",
                    confidence=0.85,
                ))
        
        # 8. 가족 구성원 → 가명처리
        # 이름 뒤 경칭으로 사용되는 경우("박순자 할머니")는 가족 참조에서 제외
        _honorific_family = {"할머니", "할아버지", "할멈"}
        name_ends = {d.end for d in detections if d.pii_type == PIIType.NAME}
        # 경칭으로 스킵된 영역을 추적하여 부분 매칭 방지
        skipped_spans = []
        # 긴 키워드부터 처리하여 "할아버지" → "아버지" 부분 매칭 방지
        sorted_keywords = sorted(FAMILY_KEYWORDS.items(), key=lambda x: -len(x[0]))
        for keyword, replacement in sorted_keywords:
            for m in re.finditer(re.escape(keyword), text):
                if not self._overlaps(detections, m.start(), m.end()):
                    # 스킵된 영역과 겹치면 무시
                    if any(m.start() >= ss and m.end() <= se for ss, se in skipped_spans):
                        continue
                    # 이름 바로 뒤에 오는 할머니/할아버지는 경칭으로 간주
                    if keyword in _honorific_family:
                        preceding_pos = m.start()
                        if any(abs(preceding_pos - ne) <= 1 for ne in name_ends):
                            skipped_spans.append((m.start(), m.end()))
                            continue
                    detections.append(PIIDetection(
                        pii_type=PIIType.FAMILY_REFERENCE,
                        original_text=keyword,
                        start=m.start(), end=m.end(),
                        masking_strategy=MaskingStrategy.PSEUDONYMIZE,
                        masked_text=replacement,
                        confidence=0.95,
                    ))
        
        # 9. 낙인 건강 상태
        # 단일 글자 조건(예: "암")은 형태소 분석으로 NNG 여부 확인하여 오탐 방지
        nng_morphs = (
            {morph for morph, tag in pos_result if tag in ("NNG", "NNP")}
            if pos_result is not None else None
        )
        for condition, replacement in STIGMATIZED_CONDITIONS.items():
            if len(condition) == 1 and nng_morphs is not None:
                if condition not in nng_morphs:
                    continue
            for m in re.finditer(re.escape(condition), text):
                if not self._overlaps(detections, m.start(), m.end()):
                    detections.append(PIIDetection(
                        pii_type=PIIType.STIGMATIZED_CONDITION,
                        original_text=condition,
                        start=m.start(), end=m.end(),
                        masking_strategy=MaskingStrategy.GENERALIZE,
                        masked_text=replacement,
                        confidence=0.95,
                    ))
        
        # 10. 정서적 민감 표현
        # bareunpy 가용 시: VA 어근 교차 매칭 (precision 향상)
        if pos_result is not None:
            va_stems = {morph for morph, tag in pos_result if tag == "VA"}
            vv_stems = {morph for morph, tag in pos_result if tag == "VV"}
            va_to_search = va_stems & set(EMOTIONAL_VA_STEMS.keys())
            # ㅂ불규칙 "~워하다" VV 파생형도 포함 (부끄러워하→부끄럽)
            for vv in vv_stems:
                if vv in _BIEUP_VV_TO_VA:
                    va_to_search.add(_BIEUP_VV_TO_VA[vv])
        else:
            va_to_search = set(EMOTIONAL_VA_STEMS.keys())

        for stem in va_to_search:
            replacement = EMOTIONAL_VA_STEMS[stem]
            for m in self._va_stem_regexes[stem].finditer(text):
                if not self._overlaps(detections, m.start(), m.end()):
                    detections.append(PIIDetection(
                        pii_type=PIIType.EMOTIONAL_SENSITIVE,
                        original_text=m.group(),
                        start=m.start(), end=m.end(),
                        masking_strategy=MaskingStrategy.GENERALIZE,
                        masked_text=replacement,
                        confidence=0.9 if pos_result is not None else 0.75,
                    ))

        for kw, replacement in EMOTIONAL_KEYWORD_STEMS.items():
            for m in self._kw_stem_regexes[kw].finditer(text):
                if not self._overlaps(detections, m.start(), m.end()):
                    detections.append(PIIDetection(
                        pii_type=PIIType.EMOTIONAL_SENSITIVE,
                        original_text=m.group(),
                        start=m.start(), end=m.end(),
                        masking_strategy=MaskingStrategy.GENERALIZE,
                        masked_text=replacement,
                        confidence=0.85,
                    ))
        
        # 위치순 정렬
        detections.sort(key=lambda d: d.start)
        return detections
    
    def redact(self, text: str) -> RedactionResult:
        """PII 감지 후 마스킹 적용
        
        Args:
            text: 입력 텍스트
        
        Returns:
            RedactionResult: 원문, 삭제된 텍스트, 감지 목록, 보존된 임상 용어
        """
        detections = self.detect_pii(text)
        
        if not detections:
            return RedactionResult(
                original_text=text,
                redacted_text=text,
                detections=[],
                pii_count=0,
                clinical_terms_preserved=self._extract_clinical_terms(text),
            )
        
        # 뒤에서부터 치환 (위치 인덱스 유지)
        redacted = text
        for det in reversed(detections):
            redacted = redacted[:det.start] + det.masked_text + redacted[det.end:]
        
        clinical_terms = self._extract_clinical_terms(redacted)
        
        return RedactionResult(
            original_text=text,
            redacted_text=redacted,
            detections=detections,
            pii_count=len(detections),
            clinical_terms_preserved=clinical_terms,
        )
    
    def generate_health_summary(self, text: str, risk_signals: list = None) -> Dict:
        """대화에서 비식별화된 건강 요약 생성 (템플릿 기반)
        
        논문 Section 5a: Template-Based Health Summary Extraction
        NER + 건강 위험 신호 결과를 구조화된 임상 형식에 매핑
        
        Args:
            text: 원문 대화 텍스트
            risk_signals: HealthRiskSignal 리스트 (전처리 결과 재사용)
        
        Returns:
            구조화된 건강 요약 dict
        """
        redaction = self.redact(text)
        
        summary = {
            "risk_level": "LOW",
            "categories": [],
            "symptoms": [],
            "recommendations": [],
            "context_redacted": redaction.redacted_text,
            "pii_removed_count": redaction.pii_count,
            "clinical_terms_preserved": redaction.clinical_terms_preserved,
        }
        
        if risk_signals:
            levels = [sig.level.value if hasattr(sig.level, 'value') else sig.level 
                      for sig in risk_signals]
            level_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            max_level = max(levels, key=lambda x: level_order.get(x, 0))
            summary["risk_level"] = max_level.upper()
            
            for sig in risk_signals:
                cat = sig.category.value if hasattr(sig.category, 'value') else sig.category
                if cat not in summary["categories"]:
                    summary["categories"].append(cat)
                summary["symptoms"].extend(sig.trigger_terms)
                if sig.recommendation and sig.recommendation not in summary["recommendations"]:
                    summary["recommendations"].append(sig.recommendation)
        
        # 중복 제거
        summary["symptoms"] = list(dict.fromkeys(summary["symptoms"]))
        
        return summary
    
    def reset_session(self):
        """새 세션 시작 시 매핑 초기화"""
        self._name_map.clear()
        self._family_map.clear()
        self._name_counter = 0
        self._family_counter = 0
    
    # ─── 내부 헬퍼 ───
    
    def _overlaps(self, detections: List[PIIDetection], start: int, end: int) -> bool:
        """기존 감지와 겹치는지 확인"""
        for d in detections:
            if start < d.end and end > d.start:
                return True
        return False
    
    def _get_name_pseudonym(self, name: str) -> str:
        """이름 → 가명 매핑 (동일 이름은 동일 가명)"""
        if name not in self._name_map:
            self._name_counter += 1
            label = chr(ord('A') + (self._name_counter - 1) % 26)
            self._name_map[name] = f"[사용자{label}]"
        return self._name_map[name]
    
    # 이름 뒤에 오는 경칭·호칭 패턴 (bareunpy 불가 시 폴백용)
    _HONORIFIC_PATTERN = re.compile(
        r'^\s*(?:어르신|할머니|할아버지|씨|님|선생|선생님|환자|간호사|의사|교수|사모님|사장|원장|실장)'
    )
    
    # 동사/부사 어근과 겹치는 고빈도 성씨 (bareunpy 불가 시 폴백용)
    _AMBIGUOUS_SURNAMES = {"하", "나", "오", "서", "고", "주", "구", "진", "노", "변", "원", "방", "성", "차", "민", "석"}
    
    def _is_name_by_pos(self, name: str, nnp_set: Optional[set], text: str, end_pos: int) -> bool:
        """이름 후보가 실제 인명인지 판별
        
        bareunpy 가용 시: NNP(고유명사) 태그 여부로 판별
        bareunpy 불가 시: 경칭 패턴 + 모호 성씨 휴리스틱 폴백
        """
        # --- bareunpy POS 분석 가용 시 ---
        if nnp_set is not None:
            return name in nnp_set
        
        # --- 폴백: 키워드 + 휴리스틱 ---
        if self._is_common_word_fallback(name):
            return False
        if len(name) == 3:
            return True
        # 2자 매치: 모호 성씨는 경칭 필요
        surname = name[0]
        if surname not in self._AMBIGUOUS_SURNAMES:
            return True
        after = text[end_pos:]
        return bool(self._HONORIFIC_PATTERN.match(after))
    
    @staticmethod
    def _is_common_word_fallback(word: str) -> bool:
        """bareunpy 불가 시 폴백: 최소한의 오탐 방지 키워드"""
        fallback_words = {
            "김치", "이상", "이것", "이런", "박수", "최근", "최고",
            "정도", "정말", "강력", "조금", "임신", "한번", "오늘",
            "오래", "서로", "신경", "권장", "안녕", "전체", "전혀",
            "홍삼", "고령", "문제", "손발", "손목", "배변", "백내장",
            "심장", "심한", "노인", "하루", "성별", "성인", "차이",
            "주사", "구강", "구토", "민감", "진료", "나이", "변비",
            "방문", "현재", "함께", "이후", "이전", "정상", "장애",
            "진단을", "우울증", "전문의", "전문가", "고관절", "고혈압",
        }
        return word in fallback_words
    
    def _extract_clinical_terms(self, text: str) -> List[str]:
        """텍스트에서 임상 용어 추출 (보존 확인용)"""
        clinical_keywords = [
            "혈압", "혈당", "통증", "수면", "불면", "어지럼", "관절",
            "골다공증", "당뇨", "심장", "호흡", "기침", "두통", "복통",
            "부종", "낙상", "균형", "기억력", "인지", "식욕", "체중",
            "운동", "산책", "복약", "약물", "수술", "검진", "진료",
        ]
        found = []
        for term in clinical_keywords:
            if term in text:
                found.append(term)
        return found
