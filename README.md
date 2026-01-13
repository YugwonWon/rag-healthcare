# 🏥 치매노인 맞춤형 헬스케어 RAG 챗봇

치매노인을 위한 개인화된 AI 돌봄 도우미입니다. RAG(Retrieval-Augmented Generation) 기술을 활용하여 따뜻하고 지속적인 케어를 제공합니다.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ 주요 기능

- 🗣️ **개인화된 대화**: 닉네임 기반으로 이전 대화를 기억하고 연속적인 케어 제공
- 💊 **복약 알림**: 약 복용 시간을 부드럽게 상기시켜 드립니다
- 📅 **일상 루틴 관리**: 식사, 산책, 취침 등 일과를 함께 관리합니다
- 🩺 **증상 모니터링**: 대화 중 건강 이상 징후를 감지합니다
- 🤖 **온디바이스 AI**: 로컬 임베딩 모델로 빠르고 안전한 처리

## 🛠️ 기술 스택

| 구성요소 | 기술 |
|---------|------|
| **백엔드** | FastAPI, Python 3.12+ |
| **LLM** | kanana 2.1B nano, Qwen 2.5 3B (Ollama) |
| **임베딩** | sentence-transformers/all-MiniLM-L6-v2 (384차원) |
| **벡터DB** | ChromaDB |
| **프론트엔드** | Gradio (HuggingFace Spaces) |
| **배포** | Google Cloud Run |

## 🎮 데모

👉 **[HuggingFace Spaces에서 체험하기](https://huggingface.co/spaces/Yugwon/rag-healthcare)**

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/YugwonWon/rag-healthcare.git
cd rag-healthcare

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
```

### 2. Ollama 설정 (선택사항)

```bash
# Ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# Qwen 2.5 모델 다운로드
ollama pull qwen2.5:3b

# Ollama 서버 시작
ollama serve
```

### 3. 서버 실행

```bash
# 개발 서버 실행
./server.sh

# 또는 직접 실행
uvicorn app.main:app --reload
```

### 4. API 테스트

```bash
# 헬스체크
curl http://localhost:8000/health

# 채팅 테스트
curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"nickname": "할머니", "message": "안녕하세요"}'
```

## 📁 프로젝트 구조

```
rag-healthcare/
├── app/                      # FastAPI 애플리케이션
│   ├── config.py             # 환경설정
│   ├── main.py               # 메인 서버
│   ├── model/                # LLM 모듈
│   │   ├── local_model.py    # Ollama/온디바이스 LLM
│   │   └── openai_model.py   # OpenAI Fallback
│   ├── retriever/            # RAG 검색기
│   ├── vector_store/         # ChromaDB 핸들러
│   └── healthcare/           # 헬스케어 도메인 모듈
│       ├── symptom_tracker.py
│       ├── medication_reminder.py
│       └── daily_routine.py
├── finetuning/               # 파인튜닝 스크립트
│   ├── prepare_dataset.py
│   ├── train_lora.py
│   └── merge_adapter.py
├── spaces/                   # HuggingFace Spaces 프론트엔드
├── data/                     # 데이터 디렉토리
├── docs/                     # 문서
└── Dockerfile                # Cloud Run 배포용
```

## 🔌 API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/health` | 서버 상태 확인 |
| `POST` | `/chat` | 채팅 메시지 처리 |
| `POST` | `/greeting` | 개인화된 인사말 생성 |
| `POST` | `/profile` | 환자 프로필 저장 |
| `GET` | `/profile/{nickname}` | 환자 프로필 조회 |
| `GET` | `/history/{nickname}` | 대화 기록 조회 |
| `POST` | `/documents` | 헬스케어 문서 추가 |
| `GET` | `/routine/{nickname}` | 일과 상태 조회 |

## 📊 개인화 기능

### 닉네임 기반 대화

```python
# 첫 대화
POST /chat
{
    "nickname": "영희할머니",
    "message": "산책 다녀올게요"
}

# 다음날 대화 - 이전 대화 기억
POST /greeting
{
    "nickname": "영희할머니"
}
# 응답: "영희할머니님, 좋은 아침이에요! 어제 산책 다녀온다 하셨는데 잘 다녀오셨나요?"
```

### 프로필 설정

```python
POST /profile
{
    "nickname": "영희할머니",
    "name": "김영희",
    "age": 82,
    "conditions": "고혈압, 경도치매",
    "emergency_contact": "010-1234-5678 (아들)"
}
```

## 🎓 파인튜닝

Qwen 2.5 3B 모델을 치매케어 대화 데이터로 파인튜닝할 수 있습니다.

```bash
# 샘플 데이터 생성
cd finetuning
python prepare_dataset.py --create-sample

# LoRA 파인튜닝
python train_lora.py \
    --train_data ../data/conversations/train_chat.jsonl \
    --output_dir ../outputs/qwen-healthcare-lora \
    --use_4bit

# 모델 병합
python merge_adapter.py merge \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --adapter ../outputs/qwen-healthcare-lora \
    --output ../outputs/qwen-healthcare-merged
```

자세한 내용은 [파인튜닝 가이드](docs/FINETUNING_GUIDE.md)를 참조하세요.

## ☁️ 배포

### Cloud Run 배포

```bash
# 환경 변수 설정
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=asia-northeast3

# 배포 실행
./deploy_cloudrun.sh
```

자세한 내용은 [Cloud Run 배포 가이드](docs/CLOUDRUN_DEPLOYMENT.md)를 참조하세요.

### HuggingFace Spaces

`spaces/` 디렉토리를 HuggingFace Spaces에 업로드하면 Gradio 프론트엔드가 배포됩니다.

```bash
# HuggingFace CLI 설치
pip install huggingface_hub

# 업로드
cd spaces
huggingface-cli upload your-username/healthcare-chatbot .
```

## 📖 문서

- [Cloud Run 배포 가이드](docs/CLOUDRUN_DEPLOYMENT.md)
- [파인튜닝 가이드](docs/FINETUNING_GUIDE.md)
- [데이터 준비 가이드](docs/DATA_PREPARATION.md)

## 🤝 기여

기여를 환영합니다! Pull Request를 보내주세요.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 💬 문의

질문이나 제안이 있으시면 Issue를 생성해주세요.

---

Made with ❤️ for dementia care
