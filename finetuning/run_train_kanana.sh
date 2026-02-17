#!/bin/bash
# Kanana 2.1B LoRA 학습 - CPU 모드
# 157개 소량 데이터 과적합 방지 최적화 설정
# GPU VRAM 부족으로 CPU 학습 (소량 데이터라 충분히 가능)
export CUDA_VISIBLE_DEVICES=""

cd /data3/yugwon/projects/rag-healthcare

echo "==========================================="
echo "  Kanana 2.1B LoRA SFT - CPU Mode"
echo "==========================================="
echo "Mode: CPU (GPU VRAM 부족)"
echo "Start: $(date)"
echo ""

# 출력 디렉토리 생성
mkdir -p finetuning/output

# 데이터 수량 확인
echo "=== 학습 데이터 ==="
wc -l finetuning/data/train_counseling.jsonl finetuning/data/val_counseling.jsonl
echo ""

# 학습 실행 (CPU 모드 - 소량 데이터라 충분히 실행 가능)
/data3/yugwon/projects/rag-healthcare/venv/bin/python \
    finetuning/train_kanana_lora.py \
    --cpu \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation 4 \
    --learning_rate 2e-5 \
    --lora_r 4 \
    --lora_alpha 8 \
    --lora_dropout 0.15 \
    --label_smoothing 0.1 \
    --max_seq_length 512 \
    --early_stopping_patience 2

echo ""
echo "End: $(date)"
echo ""
echo "=== 다음 단계 ==="
echo "1) LoRA 병합 + GGUF 변환:"
echo "   python finetuning/merge_and_convert.py \\"
echo "     --base_model kakaocorp/kanana-nano-2.1b-instruct \\"
echo "     --lora_path finetuning/output/kanana-counseling-lora \\"
echo "     --model_name kanana-counseling"
echo ""
echo "2) Ollama 등록:"
echo "   ollama create kanana-counseling -f models/Modelfile.kanana-counseling"
