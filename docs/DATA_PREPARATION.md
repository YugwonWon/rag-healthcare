# 데이터 준비 가이드

치매노인 헬스케어 RAG 챗봇을 위한 데이터 준비 가이드입니다.

## 🧠 전처리 파이프라인 (NER + N-gram)

논문 방법론에 따라 대화 입력에 대해 다음 전처리가 적용됩니다:

1. **NER (개체명 인식)**: 한국어 NER 모델(KoELECTRA)로 건강 관련 용어 태깅
2. **N-gram 추출**: 태깅된 용어 전후 5단어 컨텍스트 추출
3. **건강 위험 신호 감지**: 규칙 기반 + 의미적 유사도로 위험 수준 판단
4. **쿼리 확장**: 감지된 위험 카테고리 기반으로 RAG 검색 쿼리 향상

### 건강 위험 카테고리

| 카테고리 | 키워드 예시 | 위험 수준 |
|---------|-----------|---------|
| 만성질환 | 당뇨, 혈압, 심장 | HIGH |
| 수면 장애 | 불면, 수면제, 피곤 | MEDIUM |
| 낙상 위험 | 넘어지다, 어지럽다, 균형 | HIGH |
| 영양 문제 | 식욕, 체중, 변비 | MEDIUM |
| 통증 | 두통, 허리, 무릎 | MEDIUM |
| 정서 | 우울, 불안, 외롭다 | MEDIUM |
| 인지 | 기억, 건망증, 치매 | HIGH |
| 약물 | 복용, 부작용, 처방 | MEDIUM |
| 응급 | 가슴, 호흡, 의식 | CRITICAL |

## 📂 데이터 구조

```
data/
├── chroma/              # ChromaDB 벡터 저장소
├── raw/                 # 원본 데이터
├── healthcare_docs/     # 헬스케어 문서 (RAG용)
│   ├── dementia_care.md
│   ├── medication_guide.md
│   └── daily_routine.md
└── conversations/       # 대화 데이터 (파인튜닝용)
    ├── train_chat.jsonl
    └── val_chat.jsonl
```

## 📋 대화 데이터 형식

### 원본 형식 (raw)

```jsonl
{
  "id": "conv_001",
  "patient_info": "김영희, 82세 여성, 경도 치매, 고혈압",
  "caregiver_id": "CG001",
  "date": "2025-01-03",
  "dialogue": [
    {
      "speaker": "patient",
      "text": "오늘 약 먹었나?",
      "timestamp": "08:30:00"
    },
    {
      "speaker": "caregiver",
      "text": "네, 어르신. 오늘 아침 8시에 혈압약 드셨어요.",
      "timestamp": "08:30:15"
    }
  ]
}
```

### 학습 형식 (chat)

```jsonl
{
  "messages": [
    {"role": "system", "content": "당신은 치매노인을 돌보는 AI 도우미입니다..."},
    {"role": "user", "content": "오늘 약 먹었나?"},
    {"role": "assistant", "content": "네, 어르신. 오늘 아침 8시에 혈압약 드셨어요."}
  ]
}
```

### Instruction 형식 (Alpaca 스타일)

```jsonl
{
  "instruction": "치매노인의 질문에 친절하게 답변하세요.",
  "input": "오늘 약 먹었나?",
  "output": "네, 어르신. 오늘 아침 8시에 혈압약 드셨어요."
}
```

## 🏥 헬스케어 문서

### 문서 종류

1. **치매 케어 가이드**
   - 치매 단계별 증상
   - 대화 시 주의사항
   - 행동 대처법

2. **복약 가이드**
   - 일반적인 치매 관련 약물
   - 복약 시 주의사항
   - 부작용 정보

3. **일상 루틴 가이드**
   - 권장 일과표
   - 활동별 가이드라인

### 문서 형식

Markdown 또는 텍스트 형식:

```markdown
# 치매노인 케어 가이드

## 1. 대화 시 주의사항

- 천천히 명확하게 말하기
- 짧은 문장 사용하기
- 예/아니오로 답할 수 있는 질문하기
- 선택지 제공하기

## 2. 일상생활 지원

### 식사 지원
- 식사 시간을 규칙적으로 유지
- 조용한 환경에서 식사
...
```

## 🔧 데이터 전처리

### 샘플 데이터 생성

```bash
cd finetuning
python prepare_dataset.py --create-sample
```

### 원본 데이터 변환

```bash
python prepare_dataset.py \
    --input ../data/raw/conversations.jsonl \
    --output ../data/conversations \
    --format chat
```

### 데이터 분할 비율

- 학습(train): 90%
- 검증(val): 10%

## 📤 문서 업로드

### API를 통한 문서 추가

```bash
curl -X POST http://localhost:8000/documents \
    -H "Content-Type: application/json" \
    -d '{
        "documents": [
            "치매노인 대화 시 천천히 명확하게 말하고, 짧은 문장을 사용합니다.",
            "복약 시간을 놓치지 않도록 규칙적인 알림을 제공합니다."
        ],
        "metadatas": [
            {"source": "care_guide", "category": "communication"},
            {"source": "care_guide", "category": "medication"}
        ]
    }'
```

### 파일에서 문서 로드

```python
from app.vector_store import get_chroma_handler
import os

chroma = get_chroma_handler()

# 마크다운 파일 로드
docs_dir = "./data/healthcare_docs"
for filename in os.listdir(docs_dir):
    if filename.endswith(".md"):
        with open(os.path.join(docs_dir, filename), "r") as f:
            content = f.read()
        
        # 섹션별로 분할
        sections = content.split("\n## ")
        for section in sections:
            if section.strip():
                chroma.add_documents(
                    documents=[section],
                    metadatas=[{"source": filename}]
                )
```

## 📊 데이터 품질 체크리스트

### 대화 데이터

- [ ] 존댓말 사용 일관성
- [ ] 환자-생활지원사 역할 명확히 구분
- [ ] 적절한 대화 길이 (너무 짧거나 길지 않게)
- [ ] 개인정보 익명화
- [ ] 의학적으로 정확한 정보

### 헬스케어 문서

- [ ] 출처 명시
- [ ] 최신 정보 반영
- [ ] 이해하기 쉬운 표현
- [ ] 구조화된 형식

## 💡 데이터 수집 팁

### 대화 데이터 수집 시

1. **동의서 확보**: 환자 및 보호자 동의
2. **익명화**: 이름, 주소 등 개인정보 제거
3. **품질 검수**: 부적절한 내용 필터링
4. **다양성**: 다양한 상황과 주제 포함

### 문서 데이터 수집 시

1. **공신력 있는 출처**: 의료기관, 학회 자료
2. **저작권 확인**: 사용 가능한 라이선스
3. **정기 업데이트**: 최신 가이드라인 반영

## 🔐 개인정보 보호

### 익명화 규칙

| 원본 | 익명화 |
|-----|--------|
| 김영희 | 환자A 또는 닉네임 |
| 010-1234-5678 | [전화번호] |
| 서울시 강남구 | [주소] |
| 삼성병원 | [병원명] |

### 익명화 스크립트 예시

```python
import re

def anonymize_text(text):
    # 전화번호
    text = re.sub(r'01\d-\d{3,4}-\d{4}', '[전화번호]', text)
    # 주민번호
    text = re.sub(r'\d{6}-\d{7}', '[주민번호]', text)
    # 이메일
    text = re.sub(r'[\w.-]+@[\w.-]+\.\w+', '[이메일]', text)
    return text
```
