"""
Kanana 모델 LoRA 파인튜닝 스크립트
경량 파인튜닝으로 대화 스타일만 학습 (과적합 방지 최적화)

157개 소량 데이터 기준 최적화:
- LoRA r=4 (최소 rank로 스타일만 학습)
- 높은 dropout(0.15) + label smoothing(0.1)
- 낮은 LR(2e-5) + cosine decay + early stopping
- epoch당 ~20 steps → 3 epochs = ~60 steps

사용법:
    python train_kanana_lora.py --use_4bit --epochs 3
    python train_kanana_lora.py --cpu --epochs 3  # CPU 모드
"""

import os
import sys

# CPU 모드 체크 (--cpu 인자가 있으면 CUDA 비활성화)
if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import argparse
from pathlib import Path
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback


def load_jsonl(file_path: str) -> list[dict]:
    """JSONL 파일 로드"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_messages_kanana(messages: list[dict]) -> str:
    """
    Kanana 모델용 프롬프트 포맷팅
    Kanana는 ChatML 형식 사용
    """
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    return formatted


def main():
    parser = argparse.ArgumentParser(description="Kanana LoRA 파인튜닝")
    
    # 모델 설정
    parser.add_argument("--model_name", type=str, 
                        default="kakaocorp/kanana-nano-2.1b-instruct",
                        help="베이스 모델")
    parser.add_argument("--output_dir", type=str,
                        default="./finetuning/output/kanana-counseling-lora",
                        help="출력 디렉토리")
    
    # 데이터 설정
    parser.add_argument("--train_data", type=str,
                        default="./finetuning/data/train_counseling.jsonl",
                        help="학습 데이터 경로")
    parser.add_argument("--val_data", type=str,
                        default="./finetuning/data/val_counseling.jsonl",
                        help="검증 데이터 경로")
    
    # LoRA 설정 (경량 - 스타일만 학습, 과적합 방지)
    parser.add_argument("--lora_r", type=int, default=4, help="LoRA rank (최소값으로 스타일만 학습)")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha (r의 2배)")
    parser.add_argument("--lora_dropout", type=float, default=0.15, help="LoRA dropout (소량 데이터→높은 dropout)")
    
    # 학습 설정 (157샘플 과적합 방지 최적화)
    parser.add_argument("--epochs", type=int, default=3, help="에포크 수 (early stopping과 함께 사용)")
    parser.add_argument("--batch_size", type=int, default=2, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="학습률 (소량 데이터→낮은 LR로 과적합 방지)")
    parser.add_argument("--max_seq_length", type=int, default=512, help="최대 시퀀스 길이 (평균 443자, 512로 충분)")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="그래디언트 누적 (effective batch=8)")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="라벨 스무딩 (과적합 방지)")
    parser.add_argument("--early_stopping_patience", type=int, default=2, help="Early stopping patience")
    
    # 양자화
    parser.add_argument("--use_4bit", action="store_true", help="4비트 양자화 (QLoRA)")
    parser.add_argument("--use_8bit", action="store_true", help="8비트 양자화")
    parser.add_argument("--cpu", action="store_true", help="CPU로 학습 (양자화 없음)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Kanana LoRA 파인튜닝 시작 (과적합 방지 최적화)")
    print("=" * 60)
    print(f"  모델: {args.model_name}")
    print(f"  LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"  에포크: {args.epochs} (early stopping patience={args.early_stopping_patience})")
    print(f"  배치: {args.batch_size} × {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"  학습률: {args.learning_rate}, label smoothing: {args.label_smoothing}")
    print(f"  양자화: {'4bit' if args.use_4bit else '8bit' if args.use_8bit else 'FP32/BF16'}")
    print("=" * 60)
    
    # 1. 토크나이저 로드
    print("\n📚 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 양자화 설정
    bnb_config = None
    device_map = None  # CPU에서는 device_map 사용 안함
    
    if args.cpu:
        print("🖥️ CPU 모드로 실행...")
    elif args.use_4bit:
        print("🔧 4비트 양자화 (QLoRA) 설정...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        device_map = "auto"
    elif args.use_8bit:
        print("🔧 8비트 양자화 설정...")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        device_map = "auto"
    
    # 3. 모델 로드
    print(f"\n🤖 모델 로드 중: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float32 if args.cpu else (torch.bfloat16 if not bnb_config else None),
    )
    
    if bnb_config:
        model = prepare_model_for_kbit_training(model)
    
    # 4. LoRA 설정
    print(f"\n🔗 LoRA 어댑터 설정 (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    # 학습 가능 파라미터 출력
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 학습 가능 파라미터: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # 5. 데이터셋 로드
    print(f"\n📂 데이터셋 로드 중...")
    train_data = load_jsonl(args.train_data)
    val_data = load_jsonl(args.val_data) if Path(args.val_data).exists() else None
    
    print(f"   학습 데이터: {len(train_data)}개")
    if val_data:
        print(f"   검증 데이터: {len(val_data)}개")
    
    # 데이터셋 변환
    def format_sample(sample):
        return {"text": format_messages_kanana(sample["messages"])}
    
    train_dataset = Dataset.from_list([format_sample(s) for s in train_data])
    val_dataset = Dataset.from_list([format_sample(s) for s in val_data]) if val_data else None
    
    # 6. 학습 설정 (소량 데이터 과적합 방지 최적화)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.2,           # 높은 warmup (소량 데이터→천천히 시작)
        weight_decay=0.1,            # 강한 L2 정규화
        lr_scheduler_type="cosine",  # 자연스러운 LR 감소
        label_smoothing_factor=args.label_smoothing,  # 라벨 스무딩 (과적합 방지)
        logging_steps=5,             # epoch당 ~20 steps → 자주 로깅
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        load_best_model_at_end=True if val_dataset else False,  # 최적 체크포인트 자동 선택
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False if val_dataset else None,
        save_total_limit=3,
        fp16=False,
        bf16=not args.cpu,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
        seed=42,
        use_cpu=args.cpu,
        gradient_checkpointing=not args.cpu,  # VRAM 절약
        max_grad_norm=1.0,           # 그래디언트 클리핑
    )
    
    # 7. 트레이너 생성 (early stopping 콜백 포함)
    callbacks = []
    if val_dataset:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )
        print(f"📌 Early stopping 활성화 (patience={args.early_stopping_patience})")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    
    # 8. 학습 시작
    print("\n🎯 학습 시작...")
    trainer.train()
    
    # 9. 모델 저장
    print(f"\n💾 모델 저장 중: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # 학습 결과 출력
    train_result = trainer.state.log_history
    print("\n" + "=" * 60)
    print("✅ 파인튜닝 완료!")
    print("=" * 60)
    print(f"📁 출력 경로: {args.output_dir}")
    
    # 최종 손실 출력
    train_losses = [h['loss'] for h in train_result if 'loss' in h]
    eval_losses = [h['eval_loss'] for h in train_result if 'eval_loss' in h]
    if train_losses:
        print(f"📉 최종 train loss: {train_losses[-1]:.4f}")
    if eval_losses:
        print(f"📉 최종 eval loss: {eval_losses[-1]:.4f}")
        print(f"📉 최적 eval loss: {min(eval_losses):.4f}")
    
    print("\n다음 단계:")
    print("  1. LoRA 병합 + GGUF 변환:")
    print("     python merge_and_convert.py \\")
    print(f"       --base_model {args.model_name} \\")
    print(f"       --lora_path {args.output_dir} \\")
    print("       --model_name kanana-counseling")
    print()
    print("  2. Ollama 등록:")
    print("     ollama create kanana-counseling -f models/Modelfile.kanana-counseling")


if __name__ == "__main__":
    main()
