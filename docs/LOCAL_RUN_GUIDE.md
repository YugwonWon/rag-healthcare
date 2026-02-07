# 🏥 로컬 실행 가이드

## 현재 구조

```
맥북 (로컬)                         USB (SAMSUNG-USB)
─────────────                       ─────────────────
rag-healthcare/                     /Volumes/SAMSUNG-USB/
├── app/ (소스코드)                 ├── venv/ (파이썬 가상환경, ~2GB)
├── data/                           ├── ollama_models/ (Ollama 모델 저장소)
├── .env (설정)                     └── models/
├── venv → USB의 venv (심볼릭 링크)     └── exaone-counseling-merged/ (원본 모델)
└── ...                                 └── exaone-counseling-q8.gguf (변환된 모델)
```

---

## 🚀 실행 순서 (매번 이 순서대로!)

### 1단계: USB 연결 확인

USB를 꽂은 후 Finder에 **SAMSUNG-USB**가 보이는지 확인합니다.

터미널에서 확인:
```bash
ls /Volumes/SAMSUNG-USB/
```
`venv`, `ollama_models`, `models` 폴더가 보이면 정상입니다.

> ⚠️ USB가 안 보이면 뽑았다 다시 꽂아보세요.

---

### 2단계: Ollama 실행 확인

Ollama 앱이 실행 중인지 확인합니다. (메뉴바에 🦙 아이콘)

```bash
# Ollama 상태 확인
ollama list
```

`k-exaone-counseling` 모델이 목록에 있으면 정상입니다.

만약 Ollama가 안 켜져있으면:
- **Applications → Ollama** 앱을 더블클릭해서 실행

> ⚠️ 첫 실행 시 `OLLAMA_MODELS` 환경변수가 필요합니다.
> `~/.zshrc`에 이미 추가해둠: `export OLLAMA_MODELS=/Volumes/SAMSUNG-USB/ollama_models`

---

### 3단계: 서버 시작

프로젝트 폴더로 이동 후 서버를 실행합니다:

```bash
cd ~/Desktop/projects/rag-healthcare
./venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

아래 메시지가 나오면 성공:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
📚 컨렉션 통계: 문서=0, 대화=0, 프로필=0
```

> 💡 서버를 끄려면: `Ctrl + C`

---

### 4단계: 테스트

**서버를 켜둔 상태에서** 새 터미널 탭을 열고 (⌘+T) 테스트합니다:

#### 방법 1: 브라우저로 API 문서 보기
브라우저에서 아래 주소를 엽니다:
```
http://127.0.0.1:8000/docs
```
→ Swagger UI가 나오면 성공! 여기서 직접 API를 테스트할 수도 있습니다.

#### 방법 2: 터미널에서 curl로 테스트

```bash
# 간단한 인사 테스트
curl -s http://127.0.0.1:8000/chat \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"nickname":"테스트","message":"안녕하세요"}'
```

```bash
# 건강 상담 테스트
curl -s http://127.0.0.1:8000/chat \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"nickname":"보호자","message":"어머니가 요즘 잠을 못 주무세요"}'
```

#### 방법 3: Python으로 테스트

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/chat",
    json={"nickname": "보호자", "message": "어머니가 요즘 잠을 못 주무세요"}
)
print(response.json()["response"])
```

---

## 📋 API 엔드포인트 목록

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 서버 상태 |
| GET | `/health` | 헬스체크 |
| GET | `/docs` | API 문서 (Swagger UI) |
| POST | `/chat` | **채팅 (메인 기능)** |
| POST | `/greeting` | 인사말 생성 |
| POST | `/profile` | 환자 프로필 등록 |
| GET | `/profile/{nickname}` | 프로필 조회 |
| GET | `/history/{nickname}` | 대화 기록 조회 |
| DELETE | `/history/{nickname}` | 대화 기록 삭제 |
| POST | `/documents` | 문서 추가 |
| GET | `/stats` | 통계 |
| POST | `/medication/record` | 투약 기록 |
| GET | `/routine/{nickname}` | 일상 루틴 조회 |

---

## ❗ 문제 해결

### "venv 없음" 에러
```bash
# 심볼릭 링크 확인
ls -la venv
# → venv -> /Volumes/SAMSUNG-USB/venv 이면 정상
# USB가 연결 안 된 상태에서는 빨간 에러 뜸 → USB 꽂기
```

### "포트 이미 사용 중" 에러
```bash
# 기존 서버 프로세스 종료
lsof -ti:8000 | xargs kill -9
# 다시 서버 시작
```

### Ollama 모델 못 찾음
```bash
# 환경변수 확인
echo $OLLAMA_MODELS
# → /Volumes/SAMSUNG-USB/ollama_models 이어야 함

# 모델 목록 확인
ollama list
# → k-exaone-counseling 이 있어야 함

# 없으면 터미널을 새로 열어서 (zshrc 적용) 다시 확인
```

### 첫 요청이 느림 (10~15초)
- 정상입니다! 첫 요청 시 NER 모델(KoELECTRA)을 다운로드/로딩합니다.
- 두 번째 요청부터는 3~5초로 빨라집니다.

---

## 🔄 매일 하는 순서 요약

1. **USB 꽂기**
2. **Ollama 앱 실행** (이미 실행 중이면 생략)
3. **터미널 열기** → `cd ~/Desktop/projects/rag-healthcare`
4. **서버 시작** → `./venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`
5. **브라우저** → `http://127.0.0.1:8000/docs` 에서 테스트
6. **끝나면** → 터미널에서 `Ctrl+C`로 서버 종료
