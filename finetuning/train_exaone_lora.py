"""
EXAONE 4.0 1.2B LoRA 파인튜닝 스크립트
온디바이스 경량 모델 (1.28B 파라미터)
- QLoRA 4bit: RTX 2080 Ti 11GB에서 학습 가능
- EXAONE 프롬프트 형식: [|system|]...[|endofturn|]

사용법:
    # GPU QLoRA (권장)
    python train_exaone_lora.py --use_4bit --epochs 7
    
    # CPU 모드 (테스트용)
    python train_exaone_lora.py --cpu --epochs 1
"""

import os
import sys

# CPU 모드 체크
if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
import argparse
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig


def load_jsonl(file_path: str) -> list[dict]:
    """JSONL 파일 로드"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_messages_exaone(messages: list[dict]) -> str:
    """
    EXAONE 4.0 프롬프트 포맷팅 (비추론 모드)
    
    비추론 모드: assistant 응답에 빈 <think></think>를 붙여서
    모델이 thinking 없이 바로 답변하는 패턴을 학습시킴.
    
    형식:
      [|system|]{system_prompt}[|endofturn|]
      [|user|]{user_msg}[|endofturn|]
      [|assistant|]<think>
      </think>{assistant_msg}[|endofturn|]
    """
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "assistant":
            # 비추론 모드: 빈 thinking 블록 + 바로 답변
            formatted += f"[|{role}|]<think>\n</think>{content}[|endofturn|]\n"
        else:
            formatted += f"[|{role}|]{content}[|endofturn|]\n"
    return formatted


def main():
    parser = argparse.ArgumentParser(description="EXAONE 4.0 1.2B LoRA 파인튜닝")
    
    # 모델 설정
    parser.add_argument("--model_name", type=str,
                        default="LGAI-EXAONE/EXAONE-4.0-1.2B",
                        help="베이스 모델 (HuggingFace)")
    parser.add_argument("--output_dir", type=str,
                        default="./finetuning/output/exaone-counseling-lora",
                        help="출력 디렉토리")
    
    # 데이터 설정
    parser.add_argument("--train_data", type=str,
                        default="./finetuning/data/train_counseling.jsonl",
                        help="학습 데이터 경로")
    parser.add_argument("--val_data", type=str,
                        default="./finetuning/data/val_counseling.jsonl",
                        help="검증 데이터 경로")
    
    # LoRA 설정
    # EXAONE 1.2B는 Kanana 2.1B보다 작아서 rank를 약간 높여도 OK
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # 학습 설정
    parser.add_argument("--epochs", type=int, default=7,
                        help="에포크 수 (소량 데이터는 5-10 권장)")
    parser.add_argument("--batch_size", type=int, default=2, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="학습률 (소량 데이터는 5e-5~2e-4)")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="최대 시퀀스 길이")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                        help="그래디언트 누적 (effective batch = batch * accum)")
    
    # 양자화
    parser.add_argument("--use_4bit", action="store_true",
                        help="4비트 양자화 QLoRA (RTX 2080 Ti 11GB 권장)")
    parser.add_argument("--use_8bit", action="store_true",
                        help="8비트 양자화")
    parser.add_argument("--cpu", action="store_true",
                        help="CPU 학습 (테스트용)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 EXAONE 4.0 1.2B LoRA 파인튜닝 시작")
    print("=" * 60)
    print(f"  모델: {args.model_name}")
    print(f"  LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"  에포크: {args.epochs}")
    print(f"  배치: {args.batch_size} × {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"  학습률: {args.learning_rate}")
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
    device_map = None
    
    if args.cpu:
        print("🖥️  CPU 모드")
    elif args.use_4bit:
        print("🔧 4비트 QLoRA 설정 (RTX 2080 Ti 11GB 최적화)...")
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
    else:
        # EXAONE 1.2B FP16은 ~2.5GB → 11GB GPU에 여유 있음
        device_map = "auto"
    
    # 3. 모델 로드
    print(f"\n🤖 모델 로드 중: {args.model_name}")
    print("   (EXAONE 4.0 1.2B: 30 layers, 2048 hidden, GQA 32/8)")
    
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
    # EXAONE 4.0 아키텍처 모듈명 (transformers 모델 소스 확인):
    #   Attention: q_proj, k_proj, v_proj, o_proj  (Exaone4Attention)
    #   MLP: gate_proj, up_proj, down_proj  (Exaone4MLP / Olmo2MLP 기반 SwiGLU)
    print(f"\n🔗 LoRA 어댑터 설정 (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention
            "gate_proj", "up_proj", "down_proj",       # MLP (SwiGLU)
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    # 학습 가능 파라미터 출력
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable_params / total_params
    print(f"📊 학습 파라미터: {trainable_params:,} / {total_params:,} ({pct:.2f}%)")
    
    # 5. 데이터셋 로드
    print(f"\n📂 데이터셋 로드 중...")
    train_data = load_jsonl(args.train_data)
    val_data = load_jsonl(args.val_data) if Path(args.val_data).exists() else None
    
    print(f"   학습: {len(train_data)}개")
    if val_data:
        print(f"   검증: {len(val_data)}개")
    
    # EXAONE 형식으로 변환
    def format_sample(sample):
        return {"text": format_messages_exaone(sample["messages"])}
    
    train_dataset = Dataset.from_list([format_sample(s) for s in train_data])
    val_dataset = Dataset.from_list([format_sample(s) for s in val_data]) if val_data else None
    
    # 6. 학습 설정
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.15,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        fp16=False,
        bf16=not args.cpu,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
        seed=42,
        use_cpu=args.cpu,
        gradient_checkpointing=not args.cpu,  # VRAM 절약
    )
    
    # 7. 트레이너 생성
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    
    # 8. 학습 시작
    print("\n🎯 학습 시작...")
    trainer.train()
    
    # 9. 모델 저장
    print(f"\n💾 모델 저장 중: {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("\n" + "=" * 60)
    print("✅ EXAONE 4.0 파인튜닝 완료!")
    print("=" * 60)
    print(f"📁 출력: {args.output_dir}")
    print("\n다음 단계:")
    print("  1. LoRA 병합 + GGUF 변환:")
    print("     python merge_and_convert.py \\")
    print(f"       --base_model {args.model_name} \\")
    print(f"       --lora_path {args.output_dir} \\")
    print("       --model_name exaone-counseling")
    print()
    print("  2. Ollama 등록:")
    print("     ollama create exaone-counseling -f models/Modelfile.exaone-1.2b")
    print()
    print("  3. 테스트:")
    print("     ollama run exaone-counseling '안녕하세요, 어르신'")


if __name__ == "__main__":
    main()
