"""
LoRA/QLoRA 파인튜닝 스크립트
Qwen 2.5 3B 모델을 치매케어 대화 데이터로 파인튜닝
"""

import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


@dataclass
class ModelArguments:
    """모델 관련 인자"""
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-3B-Instruct",
        metadata={"help": "베이스 모델 경로 또는 이름"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "4비트 양자화 사용 여부 (QLoRA)"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "8비트 양자화 사용 여부"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "원격 코드 신뢰 여부"}
    )


@dataclass
class DataArguments:
    """데이터 관련 인자"""
    train_data_path: str = field(
        default="./data/conversations/train_chat.jsonl",
        metadata={"help": "학습 데이터 경로"}
    )
    val_data_path: Optional[str] = field(
        default="./data/conversations/val_chat.jsonl",
        metadata={"help": "검증 데이터 경로"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "최대 시퀀스 길이"}
    )


@dataclass
class LoraArguments:
    """LoRA 관련 인자"""
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "타겟 모듈 (쉼표로 구분)"}
    )


def load_model_and_tokenizer(model_args: ModelArguments):
    """모델과 토크나이저 로드"""
    print(f"🔄 모델 로드 중: {model_args.model_name_or_path}")
    
    # 양자화 설정
    bnb_config = None
    if model_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif model_args.use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16 if not bnb_config else None,
    )
    
    if bnb_config:
        model = prepare_model_for_kbit_training(model)
    
    print("✅ 모델 로드 완료")
    return model, tokenizer


def setup_lora(model, lora_args: LoraArguments):
    """LoRA 설정"""
    target_modules = lora_args.target_modules.split(",") if lora_args.target_modules else None
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def load_and_prepare_dataset(data_args: DataArguments, tokenizer):
    """데이터셋 로드 및 전처리"""
    print(f"📂 데이터 로드 중: {data_args.train_data_path}")
    
    def load_jsonl(file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    train_data = load_jsonl(data_args.train_data_path)
    val_data = None
    if data_args.val_data_path and os.path.exists(data_args.val_data_path):
        val_data = load_jsonl(data_args.val_data_path)
    
    def format_chat_template(example):
        """채팅 템플릿 적용"""
        messages = example.get("messages", [])
        
        # Qwen 채팅 형식으로 변환
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}
    
    def tokenize_function(examples):
        """토큰화"""
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=data_args.max_seq_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    # Dataset 생성
    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.map(format_chat_template)
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_dataset = None
    if val_data:
        val_dataset = Dataset.from_list(val_data)
        val_dataset = val_dataset.map(format_chat_template)
        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
    
    print(f"✅ 데이터셋 준비 완료: 학습 {len(train_dataset)}개, 검증 {len(val_dataset) if val_dataset else 0}개")
    
    return train_dataset, val_dataset


def train(
    model_args: ModelArguments,
    data_args: DataArguments,
    lora_args: LoraArguments,
    output_dir: str = "./outputs",
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.03,
    save_steps: int = 100,
    logging_steps: int = 10,
):
    """학습 실행"""
    # 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(model_args)
    
    # LoRA 설정
    model = setup_lora(model, lora_args)
    
    # 데이터셋 준비
    train_dataset, val_dataset = load_and_prepare_dataset(data_args, tokenizer)
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=save_steps if val_dataset else None,
        bf16=True,
        report_to="none",  # wandb 등 사용 시 변경
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )
    
    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Trainer 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 학습 시작
    print("🚀 학습 시작...")
    trainer.train()
    
    # 모델 저장
    print(f"💾 모델 저장 중: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("✅ 학습 완료!")
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA 파인튜닝")
    
    # 모델 인자
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--use_8bit", action="store_true", default=False)
    
    # 데이터 인자
    parser.add_argument("--train_data", type=str, default="./data/conversations/train_chat.jsonl")
    parser.add_argument("--val_data", type=str, default="./data/conversations/val_chat.jsonl")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    
    # LoRA 인자
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # 학습 인자
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen-healthcare-lora")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    
    args = parser.parse_args()
    
    # 인자 객체 생성
    model_args = ModelArguments(
        model_name_or_path=args.model_name,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
    )
    
    data_args = DataArguments(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        max_seq_length=args.max_seq_length,
    )
    
    lora_args = LoraArguments(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # 학습 실행
    train(
        model_args=model_args,
        data_args=data_args,
        lora_args=lora_args,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
    )
