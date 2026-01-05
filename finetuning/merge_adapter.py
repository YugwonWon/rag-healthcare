"""
LoRA 어댑터 병합 스크립트
파인튜닝된 LoRA 어댑터를 베이스 모델과 병합
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_adapter(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_model_id: str = None,
):
    """
    LoRA 어댑터와 베이스 모델 병합
    
    Args:
        base_model_path: 베이스 모델 경로
        adapter_path: LoRA 어댑터 경로
        output_path: 병합된 모델 저장 경로
        push_to_hub: HuggingFace Hub에 업로드 여부
        hub_model_id: Hub 모델 ID
    """
    print(f"🔄 베이스 모델 로드 중: {base_model_path}")
    
    # 베이스 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    
    print(f"🔄 LoRA 어댑터 로드 중: {adapter_path}")
    
    # LoRA 어댑터 적용
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("🔄 모델 병합 중...")
    
    # 어댑터 병합
    model = model.merge_and_unload()
    
    print(f"💾 병합된 모델 저장 중: {output_path}")
    
    # 병합된 모델 저장
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Hub에 업로드
    if push_to_hub and hub_model_id:
        print(f"☁️ Hub에 업로드 중: {hub_model_id}")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
    
    print("✅ 병합 완료!")


def create_ollama_modelfile(
    model_path: str,
    output_path: str,
    model_name: str = "qwen-healthcare"
):
    """
    Ollama Modelfile 생성
    
    Args:
        model_path: GGUF 모델 경로
        output_path: Modelfile 저장 경로
        model_name: 모델 이름
    """
    modelfile_content = f'''# Qwen 2.5 Healthcare 치매케어 모델
FROM {model_path}

# 파라미터 설정
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 2048

# 시스템 프롬프트
SYSTEM """당신은 치매노인을 돌보는 따뜻하고 친절한 AI 도우미입니다. 
다음 지침을 따라 대화해주세요:

1. 항상 존댓말을 사용하고, 천천히 명확하게 설명합니다.
2. 복잡한 내용은 짧고 간단한 문장으로 나눠서 전달합니다.
3. 환자의 감정을 존중하고 공감하며 대화합니다.
4. 이전 대화 내용을 자연스럽게 언급하여 연속성을 유지합니다.
5. 복약 시간, 식사, 산책 등 일상 루틴을 부드럽게 상기시킵니다.
6. 위험한 상황이나 건강 이상 징후가 감지되면 보호자/의료진 연락을 권합니다.
"""

# 템플릿 (Qwen 형식)
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
'''
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    
    print(f"✅ Modelfile 생성됨: {output_path}")
    print(f"📝 Ollama에 등록하려면:")
    print(f"   ollama create {model_name} -f {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA 어댑터 병합")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # merge 명령
    merge_parser = subparsers.add_parser("merge", help="LoRA 어댑터 병합")
    merge_parser.add_argument("--base_model", type=str, required=True, help="베이스 모델 경로")
    merge_parser.add_argument("--adapter", type=str, required=True, help="LoRA 어댑터 경로")
    merge_parser.add_argument("--output", type=str, required=True, help="출력 경로")
    merge_parser.add_argument("--push_to_hub", action="store_true", help="Hub에 업로드")
    merge_parser.add_argument("--hub_model_id", type=str, help="Hub 모델 ID")
    
    # modelfile 명령
    modelfile_parser = subparsers.add_parser("modelfile", help="Ollama Modelfile 생성")
    modelfile_parser.add_argument("--model_path", type=str, required=True, help="GGUF 모델 경로")
    modelfile_parser.add_argument("--output", type=str, default="Modelfile", help="Modelfile 경로")
    modelfile_parser.add_argument("--name", type=str, default="qwen-healthcare", help="모델 이름")
    
    args = parser.parse_args()
    
    if args.command == "merge":
        merge_lora_adapter(
            base_model_path=args.base_model,
            adapter_path=args.adapter,
            output_path=args.output,
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
        )
    elif args.command == "modelfile":
        create_ollama_modelfile(
            model_path=args.model_path,
            output_path=args.output,
            model_name=args.name,
        )
    else:
        parser.print_help()
