#!/bin/bash
# EXAONE 4.0 1.2B LoRA 학습 - GPU 3번 사용
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /data3/yugwon/projects/rag-healthcare

echo "=== EXAONE 4.0 LoRA Training ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo ""

mkdir -p finetuning/output

/data3/yugwon/projects/rag-healthcare/venv/bin/python \
    finetuning/train_exaone_lora.py \
    --use_4bit \
    --epochs 7 \
    --batch_size 1 \
    --gradient_accumulation 8 \
    --max_seq_length 512 \
    2>&1 | tee finetuning/output/train_exaone.log

echo ""
echo "End: $(date)"
