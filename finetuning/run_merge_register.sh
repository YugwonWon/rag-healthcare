#!/bin/bash
# LoRA 병합 → GGUF 변환 → Ollama 등록
export CUDA_VISIBLE_DEVICES=3
cd /data3/yugwon/projects/rag-healthcare

PYTHON=/data3/yugwon/projects/rag-healthcare/venv/bin/python
LORA_PATH=./finetuning/output/exaone-counseling-lora
BASE_MODEL=LGAI-EXAONE/EXAONE-4.0-1.2B
MERGED_DIR=./finetuning/output/exaone-counseling-merged
GGUF_OUT=./models/exaone-counseling-finetuned.gguf

echo "=== Step 1: LoRA 병합 ==="
echo "Start: $(date)"

$PYTHON -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

lora_path = '$LORA_PATH'
base_model = '$BASE_MODEL'
output_dir = '$MERGED_DIR'

print('📚 토크나이저 로드...')
tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)

print('🤖 베이스 모델 로드...')
base = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)

print('🔗 LoRA 병합...')
model = PeftModel.from_pretrained(base, lora_path)
model = model.merge_and_unload()

print(f'💾 저장: {output_dir}')
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
print('✅ Step 1 완료: 병합 완료!')
"

if [ $? -ne 0 ]; then
    echo "❌ 병합 실패"
    exit 1
fi

echo ""
echo "=== Step 2: GGUF 변환 ==="
echo "$(date)"

# llama.cpp 경로 탐색
LLAMA_CPP=""
for p in ~/llama.cpp /opt/llama.cpp /usr/local/llama.cpp; do
    if [ -f "$p/convert_hf_to_gguf.py" ]; then
        LLAMA_CPP=$p
        break
    fi
done

if [ -z "$LLAMA_CPP" ]; then
    echo "⚠️ llama.cpp 없음 - pip install llama-cpp-python 시도"
    
    # transformers + gguf 패키지로 변환 시도
    $PYTHON -c "
import subprocess, sys
# gguf 패키지가 있으면 convert_hf_to_gguf.py 대신 사용
try:
    import gguf
    print('gguf 패키지 발견:', gguf.__version__)
except ImportError:
    print('gguf 패키지 설치 중...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gguf', '-q'])
"
    
    # huggingface에서 직접 GGUF export 시도 (transformers >= 4.44)
    $PYTHON -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print('🔄 GGUF 변환 시도 (transformers export)...')
model_path = '$MERGED_DIR'
output_path = '$GGUF_OUT'

try:
    from transformers.gguf import convert_to_gguf
    convert_to_gguf(model_path, output_path, quantization='q4_k_m')
    print('✅ GGUF 변환 완료:', output_path)
except (ImportError, AttributeError) as e:
    print(f'⚠️ transformers GGUF export 불가: {e}')
    print('💡 대안: llama-cpp-python으로 변환합니다')
    
    # 최후 수단: llama-cpp-python convert
    import subprocess, sys
    try:
        result = subprocess.run([
            sys.executable, '-m', 'llama_cpp.convert',
            model_path, '--outfile', output_path, '--outtype', 'q4_k_m'
        ], capture_output=True, text=True)
        if result.returncode == 0:
            print('✅ GGUF 변환 완료:', output_path)
        else:
            print('❌ 변환 실패:', result.stderr[:500])
            print('')
            print('수동 변환이 필요합니다:')
            print('  pip install llama-cpp-python')
            print(f'  python -m llama_cpp.convert {model_path} --outfile {output_path} --outtype q4_k_m')
    except Exception as e2:
        print(f'❌ llama-cpp-python도 없음: {e2}')
        print('병합된 모델은 여기 있습니다:', model_path)
        sys.exit(1)
"
else
    echo "llama.cpp 발견: $LLAMA_CPP"
    FP16_GGUF=${GGUF_OUT%.gguf}-fp16.gguf
    
    $PYTHON $LLAMA_CPP/convert_hf_to_gguf.py $MERGED_DIR --outfile $FP16_GGUF --outtype f16
    
    if [ -f "$LLAMA_CPP/build/bin/llama-quantize" ]; then
        $LLAMA_CPP/build/bin/llama-quantize $FP16_GGUF $GGUF_OUT Q4_K_M
        rm -f $FP16_GGUF
        echo "✅ GGUF 양자화 완료: $GGUF_OUT"
    else
        mv $FP16_GGUF $GGUF_OUT
        echo "✅ GGUF (FP16) 완료: $GGUF_OUT"
    fi
fi

echo ""
echo "=== Step 3: Ollama 등록 ==="
echo "$(date)"

# GGUF 파일이 있으면 등록, 없으면 기존 base GGUF 사용
if [ -f "$GGUF_OUT" ]; then
    GGUF_PATH=$(realpath $GGUF_OUT)
else
    echo "⚠️ 파인튜닝 GGUF 없음, 기존 base GGUF 사용"
    GGUF_PATH=$(realpath ./models/EXAONE-4.0-1.2B-Q4_K_M.gguf)
fi

echo "GGUF 경로: $GGUF_PATH"

# Modelfile 생성
cat > ./models/Modelfile.exaone-counseling << 'MODELFILE_END'
FROM GGUF_PLACEHOLDER

SYSTEM "당신은 노인건강전문상담사입니다. 반드시 한국어로만 응답하세요. 한자를 사용하지 마세요. 2~3문장으로 간결하게 답변하고, 공감 후 질문으로 문제를 파악하세요. 일상에서 실천할 수 있는 건강 습관을 안내하고, 심각한 경우에만 병원 진료를 권유하세요."

PARAMETER temperature 0.1
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER num_predict 256
PARAMETER repeat_penalty 1.1
PARAMETER stop "[|endofturn|]"

TEMPLATE """{{- if .System }}[|system|]{{ .System }}[|endofturn|]
{{- end }}{{- range .Messages }}
{{- if eq .Role "user" }}[|user|]{{ .Content }}[|endofturn|]
{{- else if eq .Role "assistant" }}[|assistant|]{{ .Content }}[|endofturn|]
{{- end }}{{- end }}[|assistant|]<think>
</think>"""
MODELFILE_END

# GGUF 경로 치환
sed -i "s|GGUF_PLACEHOLDER|$GGUF_PATH|g" ./models/Modelfile.exaone-counseling

echo "Modelfile 내용:"
cat ./models/Modelfile.exaone-counseling

echo ""
echo "Ollama 등록 중..."
ollama create exaone-counseling -f ./models/Modelfile.exaone-counseling

echo ""
echo "=== 완료 ==="
echo "End: $(date)"
