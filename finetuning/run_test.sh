#!/bin/bash
# 독립적으로 실행되는 테스트 스크립트
# 사용법: bash finetuning/run_test.sh
# 결과: finetuning/test_output.txt

cd /data3/yugwon/projects/rag-healthcare

OUT="finetuning/test_output.txt"

echo "=== Ollama 테스트 시작: $(date) ===" > "$OUT"

# 1. 서버 상태 확인
echo "" >> "$OUT"
echo "--- Ollama 상태 ---" >> "$OUT"
curl -s http://localhost:11434/api/tags | python3 -c "
import sys,json
d=json.load(sys.stdin)
for m in d.get('models',[]):
    print(f\"  {m['name']} ({m.get('size',0)//1024//1024}MB)\")
" >> "$OUT" 2>&1

# 2. GPU 상태
echo "" >> "$OUT"
echo "--- GPU 상태 ---" >> "$OUT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader >> "$OUT" 2>&1

# 3. 테스트
PROMPTS=(
    "안녕하세요, 요즘 무릎이 시리고 아파요. 어떻게 하면 좋을까요?"
    "밤에 잠이 잘 안 와서 걱정이에요. 어떻게 해야 할까요?"
    "치매 예방하려면 어떻게 해야 하나요?"
    "혈압약을 매일 먹는데 가끐 깜빡해요. 어떡하죠?"
)
TITLES=("무릎통증" "불면증" "치매예방" "혈압약복용")

for i in 0 1 2 3; do
    echo "" >> "$OUT"
    echo "--- 테스트 $((i+1)): ${TITLES[$i]} ---" >> "$OUT"
    echo "질문: ${PROMPTS[$i]}" >> "$OUT"
    START=$(date +%s)
    
    RESP=$(curl -s --max-time 600 http://localhost:11434/api/generate \
        -d "{\"model\":\"exaone-counseling\",\"prompt\":\"${PROMPTS[$i]}\",\"stream\":false}" 2>&1)
    
    END=$(date +%s)
    ELAPSED=$((END - START))
    
    echo "소요: ${ELAPSED}초" >> "$OUT"
    echo "$RESP" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    print('답변:', d.get('response','NO_RESPONSE'))
except:
    print('오류: JSON 파싱 실패')
" >> "$OUT" 2>&1
done

echo "" >> "$OUT"
echo "=== 테스트 완료: $(date) ===" >> "$OUT"
