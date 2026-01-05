# 파인튜닝 가이드

Qwen 2.5 3B 모델을 치매노인-생활지원사 대화 데이터로 파인튜닝하는 가이드입니다.

## 📋 요구사항

### 하드웨어

- **GPU**: NVIDIA GPU (최소 16GB VRAM 권장)
  - RTX 3090/4090, A100, H100 등
  - QLoRA 사용 시 8GB VRAM으로도 가능
- **RAM**: 32GB 이상 권장
- **Storage**: 50GB 이상 여유 공간

### 소프트웨어

```bash
# CUDA 11.8 이상
nvidia-smi

# 파인튜닝 의존성 설치
cd finetuning
pip install -r requirements.txt
```

## 📂 데이터 준비

### 1. 데이터 형식

대화 데이터는 JSONL 형식으로 준비합니다:

```jsonl
{
  "id": "conv_001",
  "patient_info": "80세 여성, 경도 치매",
  "dialogue": [
    {"speaker": "patient", "text": "오늘 약 먹었나?"},
    {"speaker": "caregiver", "text": "네, 어르신. 아침에 드셨어요."}
  ]
}
```

### 2. 샘플 데이터 생성

```bash
python prepare_dataset.py --create-sample
```

### 3. 실제 데이터 변환

```bash
python prepare_dataset.py \
    --input ./data/raw/conversations.jsonl \
    --output ./data/conversations \
    --format chat
```

출력 파일:
- `train_chat.jsonl`: 학습 데이터
- `val_chat.jsonl`: 검증 데이터

## 🚀 파인튜닝 실행

### 기본 실행 (QLoRA)

```bash
python train_lora.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --train_data ./data/conversations/train_chat.jsonl \
    --val_data ./data/conversations/val_chat.jsonl \
    --output_dir ./outputs/qwen-healthcare-lora \
    --num_epochs 3 \
    --batch_size 4 \
    --use_4bit
```

### 고급 설정

```bash
python train_lora.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --train_data ./data/conversations/train_chat.jsonl \
    --output_dir ./outputs/qwen-healthcare-lora \
    --num_epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lora_r 32 \
    --lora_alpha 64 \
    --use_4bit
```

### LoRA 파라미터 설명

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `lora_r` | 16 | LoRA rank (높을수록 표현력↑, 메모리↑) |
| `lora_alpha` | 32 | LoRA 스케일링 (보통 r의 2배) |
| `lora_dropout` | 0.05 | 드롭아웃 비율 |
| `target_modules` | 전체 | 적용할 레이어 |

## 📊 모델 평가

### 테스트 데이터 평가

```bash
python eval_model.py \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --adapter_path ./outputs/qwen-healthcare-lora \
    --test_data ./data/conversations/val_chat.jsonl \
    --output eval_results.json \
    --use_4bit
```

### 대화형 평가

```bash
python eval_model.py \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --adapter_path ./outputs/qwen-healthcare-lora \
    --interactive \
    --use_4bit
```

## 🔀 어댑터 병합

### LoRA 어댑터 병합

```bash
python merge_adapter.py merge \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --adapter ./outputs/qwen-healthcare-lora \
    --output ./outputs/qwen-healthcare-merged
```

### HuggingFace Hub 업로드

```bash
python merge_adapter.py merge \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --adapter ./outputs/qwen-healthcare-lora \
    --output ./outputs/qwen-healthcare-merged \
    --push_to_hub \
    --hub_model_id your-username/qwen-healthcare
```

## 🖥️ Ollama 연동

### GGUF 변환 (llama.cpp 사용)

```bash
# llama.cpp 설치
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# 변환
python convert-hf-to-gguf.py ../outputs/qwen-healthcare-merged \
    --outfile qwen-healthcare.gguf \
    --outtype q4_k_m
```

### Ollama Modelfile 생성

```bash
python merge_adapter.py modelfile \
    --model_path ./qwen-healthcare.gguf \
    --output Modelfile \
    --name qwen-healthcare
```

### Ollama에 등록

```bash
ollama create qwen-healthcare -f Modelfile
ollama run qwen-healthcare
```

## 💡 팁 & 트릭

### 메모리 부족 시

1. `--use_4bit` 옵션 사용
2. `--batch_size` 줄이기
3. `--gradient_accumulation_steps` 늘리기

### 학습 모니터링

```bash
# Tensorboard 실행
tensorboard --logdir ./outputs/qwen-healthcare-lora

# 또는 Wandb 사용
pip install wandb
wandb login
# train_lora.py에서 report_to="wandb" 설정
```

### 체크포인트에서 재개

```bash
python train_lora.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --train_data ./data/conversations/train_chat.jsonl \
    --output_dir ./outputs/qwen-healthcare-lora \
    --resume_from_checkpoint
```

## 📈 예상 학습 시간

| GPU | 데이터 크기 | QLoRA | 예상 시간 |
|-----|------------|-------|----------|
| RTX 3090 | 1,000 샘플 | ✅ | ~30분 |
| RTX 3090 | 10,000 샘플 | ✅ | ~5시간 |
| A100 40GB | 10,000 샘플 | ❌ | ~3시간 |

## 🐛 트러블슈팅

### CUDA Out of Memory

```bash
# 배치 크기 줄이기
--batch_size 1 --gradient_accumulation_steps 16

# 더 공격적인 양자화
--use_4bit
```

### 학습이 수렴하지 않음

- Learning rate 조정: `1e-4` → `2e-5`
- LoRA rank 늘리기: `16` → `32`
- 더 많은 에폭: `3` → `5`
