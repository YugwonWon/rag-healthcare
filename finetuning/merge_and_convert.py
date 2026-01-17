"""
LoRA 어댑터 병합 및 GGUF 변환 스크립트
파인튜닝된 Kanana 모델을 Ollama에서 사용할 수 있도록 변환

사용법:
    python merge_and_convert.py --lora_path ./finetuning/output/kanana-counseling-lora
"""

import os
import argparse
import subprocess
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_adapter(base_model: str, lora_path: str, output_path: str):
    """LoRA 어댑터를 베이스 모델에 병합"""
    print(f"\n🔗 LoRA 어댑터 병합 중...")
    print(f"   베이스 모델: {base_model}")
    print(f"   LoRA 경로: {lora_path}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    
    # 베이스 모델 로드
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # LoRA 어댑터 로드 및 병합
    model = PeftModel.from_pretrained(base, lora_path)
    model = model.merge_and_unload()
    
    # 저장
    print(f"💾 병합된 모델 저장: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    print("✅ 병합 완료!")
    return output_path


def convert_to_gguf(model_path: str, output_path: str, quantization: str = "q4_k_m"):
    """
    HuggingFace 모델을 GGUF 형식으로 변환
    llama.cpp 필요
    """
    print(f"\n🔄 GGUF 변환 중 (양자화: {quantization})...")
    
    # llama.cpp 경로 확인
    llama_cpp_path = os.environ.get("LLAMA_CPP_PATH", "~/llama.cpp")
    llama_cpp_path = os.path.expanduser(llama_cpp_path)
    
    convert_script = Path(llama_cpp_path) / "convert_hf_to_gguf.py"
    quantize_bin = Path(llama_cpp_path) / "build" / "bin" / "llama-quantize"
    
    if not convert_script.exists():
        print(f"⚠️ llama.cpp를 찾을 수 없습니다: {llama_cpp_path}")
        print("   다음 명령으로 설치하세요:")
        print("   git clone https://github.com/ggerganov/llama.cpp")
        print("   cd llama.cpp && make -j")
        return None
    
    # FP16 GGUF 변환
    fp16_path = output_path.replace(".gguf", "-fp16.gguf")
    
    cmd_convert = [
        "python", str(convert_script),
        model_path,
        "--outfile", fp16_path,
        "--outtype", "f16"
    ]
    
    print(f"   실행: {' '.join(cmd_convert)}")
    result = subprocess.run(cmd_convert, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ 변환 실패: {result.stderr}")
        return None
    
    # 양자화
    if quantization != "f16":
        cmd_quantize = [
            str(quantize_bin),
            fp16_path,
            output_path,
            quantization.upper()
        ]
        
        print(f"   양자화: {' '.join(cmd_quantize)}")
        result = subprocess.run(cmd_quantize, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 양자화 실패: {result.stderr}")
            return fp16_path
        
        # FP16 파일 삭제
        os.remove(fp16_path)
    else:
        output_path = fp16_path
    
    print(f"✅ GGUF 변환 완료: {output_path}")
    return output_path


def create_ollama_modelfile(gguf_path: str, output_path: str, model_name: str):
    """Ollama Modelfile 생성"""
    modelfile_content = f'''# Kanana 상담 모델 - 파인튜닝됨
FROM {gguf_path}

# 시스템 프롬프트
SYSTEM """당신은 노인건강전문상담사입니다.
- 2~3문장으로 간결하게 답변
- 공감 후 질문으로 문제를 파악
- 일상에서 실천할 수 있는 건강 습관 안내
- 심각한 경우에만 병원 진료 권유"""

# 파라미터 설정
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 256
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"

# 템플릿 (ChatML 형식)
TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
"""
'''
    
    modelfile_path = Path(output_path) / "Modelfile"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    
    print(f"✅ Modelfile 생성: {modelfile_path}")
    print(f"\n📝 Ollama 등록 명령어:")
    print(f"   ollama create {model_name} -f {modelfile_path}")
    
    return modelfile_path


def main():
    parser = argparse.ArgumentParser(description="LoRA 병합 및 GGUF 변환")
    
    parser.add_argument("--base_model", type=str,
                        default="kakaocorp/kanana-nano-2.1b-instruct",
                        help="베이스 모델")
    parser.add_argument("--lora_path", type=str,
                        default="./finetuning/output/kanana-counseling-lora",
                        help="LoRA 어댑터 경로")
    parser.add_argument("--output_dir", type=str,
                        default="./finetuning/output/kanana-counseling-merged",
                        help="병합 모델 출력 경로")
    parser.add_argument("--quantization", type=str,
                        default="q4_k_m",
                        choices=["f16", "q8_0", "q4_k_m", "q4_k_s", "q5_k_m"],
                        help="GGUF 양자화 타입")
    parser.add_argument("--model_name", type=str,
                        default="kanana-counseling",
                        help="Ollama 모델 이름")
    parser.add_argument("--skip_merge", action="store_true",
                        help="병합 단계 건너뛰기 (이미 병합된 경우)")
    parser.add_argument("--skip_gguf", action="store_true",
                        help="GGUF 변환 건너뛰기")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 모델 변환 시작")
    print("=" * 60)
    
    merged_path = args.output_dir
    
    # 1. LoRA 병합
    if not args.skip_merge:
        merged_path = merge_lora_adapter(
            args.base_model,
            args.lora_path,
            args.output_dir
        )
    
    # 2. GGUF 변환
    gguf_path = None
    if not args.skip_gguf:
        gguf_filename = f"{args.model_name}-{args.quantization}.gguf"
        gguf_path = str(Path(args.output_dir) / gguf_filename)
        gguf_path = convert_to_gguf(merged_path, gguf_path, args.quantization)
    
    # 3. Ollama Modelfile 생성
    if gguf_path:
        create_ollama_modelfile(gguf_path, args.output_dir, args.model_name)
    
    print("\n" + "=" * 60)
    print("✅ 변환 완료!")
    print("=" * 60)
    
    if gguf_path:
        print(f"\n📁 GGUF 파일: {gguf_path}")
        print("\n🔧 Ollama 등록:")
        print(f"   cd {args.output_dir}")
        print(f"   ollama create {args.model_name} -f Modelfile")
        print(f"\n🧪 테스트:")
        print(f"   ollama run {args.model_name} '안녕하세요'")


if __name__ == "__main__":
    main()
