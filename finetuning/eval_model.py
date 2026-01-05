"""
모델 평가 스크립트
파인튜닝된 모델 성능 평가
"""

import argparse
import json
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


def load_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    use_4bit: bool = False,
):
    """모델 로드"""
    print(f"🔄 모델 로드 중: {model_path}")
    
    quantization_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not quantization_config else None,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    if adapter_path:
        print(f"🔄 어댑터 로드 중: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
):
    """응답 생성"""
    messages = [
        {"role": "system", "content": "당신은 치매노인을 돌보는 따뜻하고 친절한 AI 도우미입니다."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def evaluate_on_test_set(
    model,
    tokenizer,
    test_data_path: str,
    output_path: str,
):
    """테스트 세트에서 평가"""
    print(f"📂 테스트 데이터 로드: {test_data_path}")
    
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    
    results = []
    
    for item in tqdm(test_data, desc="평가 중"):
        messages = item.get("messages", [])
        
        # 마지막 assistant 응답 전까지의 대화 추출
        prompt_messages = []
        expected_response = None
        
        for msg in messages:
            if msg["role"] == "assistant" and expected_response is None:
                expected_response = msg["content"]
            else:
                prompt_messages.append(msg)
        
        if not expected_response:
            continue
        
        # user 메시지 추출
        user_message = next((m["content"] for m in reversed(prompt_messages) if m["role"] == "user"), None)
        if not user_message:
            continue
        
        # 응답 생성
        generated_response = generate_response(model, tokenizer, user_message)
        
        results.append({
            "input": user_message,
            "expected": expected_response,
            "generated": generated_response,
        })
    
    # 결과 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 평가 결과 저장: {output_path}")
    print(f"   총 {len(results)}개 샘플 평가됨")
    
    return results


def interactive_eval(model, tokenizer):
    """대화형 평가"""
    print("\n🗣️ 대화형 평가 모드 (종료하려면 'quit' 입력)")
    print("-" * 50)
    
    while True:
        user_input = input("\n👤 사용자: ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("👋 종료합니다.")
            break
        
        if not user_input:
            continue
        
        response = generate_response(model, tokenizer, user_input)
        print(f"\n🤖 AI: {response}")


def calculate_metrics(results: list[dict]) -> dict:
    """간단한 메트릭 계산"""
    from collections import Counter
    
    # 응답 길이 통계
    gen_lengths = [len(r["generated"]) for r in results]
    exp_lengths = [len(r["expected"]) for r in results]
    
    # 키워드 일치율 (간단한 평가)
    keyword_matches = 0
    important_keywords = ["어르신", "네", "드세요", "좋아요", "괜찮아요"]
    
    for r in results:
        gen = r["generated"].lower()
        for keyword in important_keywords:
            if keyword in gen:
                keyword_matches += 1
                break
    
    return {
        "num_samples": len(results),
        "avg_generated_length": sum(gen_lengths) / len(gen_lengths) if gen_lengths else 0,
        "avg_expected_length": sum(exp_lengths) / len(exp_lengths) if exp_lengths else 0,
        "keyword_match_rate": keyword_matches / len(results) if results else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모델 평가")
    
    parser.add_argument("--model_path", type=str, required=True, help="모델 경로")
    parser.add_argument("--adapter_path", type=str, help="LoRA 어댑터 경로")
    parser.add_argument("--test_data", type=str, help="테스트 데이터 경로")
    parser.add_argument("--output", type=str, default="eval_results.json", help="결과 저장 경로")
    parser.add_argument("--interactive", action="store_true", help="대화형 평가 모드")
    parser.add_argument("--use_4bit", action="store_true", help="4비트 양자화 사용")
    
    args = parser.parse_args()
    
    # 모델 로드
    model, tokenizer = load_model(
        args.model_path,
        args.adapter_path,
        args.use_4bit,
    )
    
    if args.interactive:
        interactive_eval(model, tokenizer)
    elif args.test_data:
        results = evaluate_on_test_set(
            model, tokenizer,
            args.test_data,
            args.output,
        )
        metrics = calculate_metrics(results)
        print("\n📊 평가 메트릭:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
    else:
        print("--test_data 또는 --interactive 옵션을 지정하세요.")
