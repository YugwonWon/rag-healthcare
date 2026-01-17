# Kanana 모델 파인튜닝 가이드

## 개요

Kanana-nano 2.1B 모델을 노인 건강상담 대화 스타일로 경량 파인튜닝(LoRA)합니다.

**목표:**
- 대화 스타일만 학습 (지식이 아닌 응답 패턴)
- 2~3문장의 간결한 응답
- 공감 → 질문 → 정보제공 패턴

## 요구사항

### 하드웨어
- **GPU**: VRAM 8GB 이상 (RTX 3070 이상)
- **RAM**: 16GB 이상

### 소프트웨어
```bash
# 파인튜닝 환경 설치
pip install -r finetuning/requirements.txt

# GGUF 변환용 llama.cpp (선택)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j
```

## 파인튜닝 워크플로우

### 1. 데이터셋 준비

```bash
# 대화 예제 파일을 학습용 JSONL로 변환
python finetuning/prepare_counseling_data.py \
    --input_dir ./data/conversations \
    --output_dir ./finetuning/data \
    --max_turns 3

# 결과:
# - finetuning/data/train_counseling.jsonl
# - finetuning/data/val_counseling.jsonl
```

**데이터 형식 예시** (`train_counseling.jsonl`):
```json
{
  "messages": [
    {"role": "system", "content": "당신은 노인건강전문상담사입니다..."},
    {"role": "user", "content": "요즘 잠을 못 자겠어요."},
    {"role": "assistant", "content": "잠을 못 주무셔서 많이 피곤하시겠어요. 하루에 몇 시간 정도 주무시나요?"}
  ]
}
```

### 2. LoRA 파인튜닝

```bash
# 경량 파인튜닝 (r=8, 약 0.5%만 학습)
python finetuning/train_kanana_lora.py \
    --epochs 3 \
    --batch_size 2 \
    --lora_r 8 \
    --learning_rate 2e-4

# GPU 메모리 부족시
python finetuning/train_kanana_lora.py \
    --epochs 3 \
    --batch_size 1 \
    --gradient_accumulation 8 \
    --lora_r 4
```

**LoRA 설정 가이드:**
| 설정 | r=4 | r=8 (권장) | r=16 |
|------|-----|------------|------|
| 학습 파라미터 | ~0.3% | ~0.5% | ~1% |
| VRAM 사용량 | ~6GB | ~8GB | ~12GB |
| 학습 시간 | 빠름 | 보통 | 느림 |
| 스타일 변화 | 약함 | 적절 | 강함 |

### 3. 모델 병합 및 GGUF 변환

```bash
# LoRA 어댑터 병합 + GGUF 변환
python finetuning/merge_and_convert.py \
    --lora_path ./finetuning/output/kanana-counseling-lora \
    --quantization q4_k_m

# 결과:
# - ./finetuning/output/kanana-counseling-merged/kanana-counseling-q4_k_m.gguf
# - ./finetuning/output/kanana-counseling-merged/Modelfile
```

### 4. Ollama 등록

```bash
cd ./finetuning/output/kanana-counseling-merged

# Ollama에 모델 등록
ollama create kanana-counseling -f Modelfile

# 테스트
ollama run kanana-counseling "요즘 잠을 못 자겠어요"
```

### 5. 애플리케이션에서 사용

`.env` 파일 수정:
```env
OLLAMA_MODEL=kanana-counseling
```

## Cloud Run 배포

### 방법 1: HuggingFace Hub에 업로드

```bash
# HuggingFace에 GGUF 업로드
huggingface-cli upload your-username/kanana-counseling-gguf \
    ./finetuning/output/kanana-counseling-merged/kanana-counseling-q4_k_m.gguf

# Dockerfile에서 다운로드하도록 수정
```

### 방법 2: Cloud Storage에 업로드

```bash
# GCS에 모델 업로드
gsutil cp ./finetuning/output/kanana-counseling-merged/kanana-counseling-q4_k_m.gguf \
    gs://your-bucket/models/

# 시작 스크립트에서 다운로드
```

### 방법 3: Docker 이미지에 포함

`cloudbuild.yaml` 수정:
```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/healthcare-rag:latest', '.']
    # GGUF 파일을 이미지에 포함
```

## 파일 구조

```
finetuning/
├── data/
│   ├── train_counseling.jsonl    # 학습 데이터
│   └── val_counseling.jsonl      # 검증 데이터
├── output/
│   ├── kanana-counseling-lora/   # LoRA 어댑터
│   └── kanana-counseling-merged/ # 병합된 모델 + GGUF
├── prepare_counseling_data.py    # 데이터 준비
├── train_kanana_lora.py          # LoRA 학습
├── merge_and_convert.py          # 병합 & GGUF 변환
└── requirements.txt              # 의존성
```

## 문제 해결

### CUDA Out of Memory
```bash
# 배치 크기 줄이기
--batch_size 1 --gradient_accumulation 8

# LoRA rank 줄이기
--lora_r 4
```

### llama.cpp 없음
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j
export LLAMA_CPP_PATH=$(pwd)
```

### 모델이 한국어를 이상하게 생성
- 에포크 수 줄이기 (overfitting 방지)
- 데이터셋 품질 확인
- LoRA rank 줄이기 (r=4)

## 참고

- [Kanana 모델](https://huggingface.co/kakaocorp/kanana-nano-2.1b-instruct)
- [PEFT 라이브러리](https://github.com/huggingface/peft)
- [llama.cpp GGUF](https://github.com/ggerganov/llama.cpp)
