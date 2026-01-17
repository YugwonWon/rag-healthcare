"""
상담 대화 예제를 파인튜닝용 데이터셋으로 변환
수면 상담 등 텍스트 형식의 대화를 JSONL 형식으로 변환
"""

import json
import re
from pathlib import Path
from typing import Optional


# 시스템 프롬프트 (대화 스타일 학습용 - 간결하게)
SYSTEM_PROMPT = """당신은 노인건강전문상담사입니다.
- 2~3문장으로 간결하게 답변
- 공감 후 질문으로 문제를 파악
- 일상에서 실천할 수 있는 건강 습관 안내
- 심각한 경우에만 병원 진료 권유"""


def parse_conversation_file(file_path: str) -> list[dict]:
    """
    텍스트 형식의 대화 파일을 파싱
    
    지원 형식:
    - "User: 메시지" / "Assistant: 메시지"
    - "고령자: 메시지" / "상담사: 메시지"
    - "Agent: 메시지"
    """
    conversations = []
    current_conv = {"system": None, "turns": []}
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 대화 블록 분리 (빈 줄 2개 이상으로 구분)
    blocks = re.split(r'\n\s*\n', content)
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        lines = block.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # System 프롬프트 감지
            if line.startswith("System") and "페르소나" in line:
                # 새 대화 시작
                if current_conv["turns"]:
                    conversations.append(current_conv)
                current_conv = {"system": SYSTEM_PROMPT, "turns": []}
                continue
            
            # User/고령자 턴
            user_match = re.match(r'^(User|고령자|이용자)\s*[:\uff1a]\s*(.+)', line)
            if user_match:
                content = user_match.group(2).strip().strip('"')
                if content:
                    current_conv["turns"].append({
                        "role": "user",
                        "content": content
                    })
                continue
            
            # Assistant/상담사/Agent 턴
            assistant_match = re.match(r'^(Assistant|Agent|상담사)\s*[:\uff1a]\s*(.+)', line)
            if assistant_match:
                content = assistant_match.group(2).strip().strip('"')
                # [대괄호 안의 지시문] 제거 but 내용은 유지
                content = re.sub(r'\[([^\]]+)\]', r'(\1)', content)
                if content:
                    current_conv["turns"].append({
                        "role": "assistant",
                        "content": content
                    })
                continue
    
    # 마지막 대화 추가
    if current_conv["turns"]:
        conversations.append(current_conv)
    
    return conversations


def convert_to_chat_format(conversations: list[dict]) -> list[dict]:
    """
    대화를 Kanana/Qwen 학습 형식으로 변환
    
    출력 형식:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    dataset = []
    
    for conv in conversations:
        messages = []
        
        # 시스템 프롬프트
        system = conv.get("system", SYSTEM_PROMPT)
        messages.append({"role": "system", "content": system})
        
        # 대화 턴
        turns = conv.get("turns", [])
        
        # user-assistant 쌍만 추출
        i = 0
        while i < len(turns):
            if turns[i]["role"] == "user":
                user_msg = turns[i]["content"]
                
                # 다음 assistant 응답 찾기
                if i + 1 < len(turns) and turns[i + 1]["role"] == "assistant":
                    assistant_msg = turns[i + 1]["content"]
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        # 최소 1쌍의 대화가 있어야 함
        if len(messages) >= 3:  # system + user + assistant
            dataset.append({"messages": messages})
    
    return dataset


def split_multi_turn_conversations(dataset: list[dict], max_turns: int = 4) -> list[dict]:
    """
    긴 대화를 여러 샘플로 분할 (슬라이딩 윈도우)
    max_turns: 최대 user-assistant 쌍 수
    """
    expanded = []
    
    for sample in dataset:
        messages = sample["messages"]
        system = messages[0]  # system prompt
        turns = messages[1:]  # user/assistant turns
        
        # 2개씩 (user, assistant) 쌍으로 그룹화
        pairs = []
        for i in range(0, len(turns) - 1, 2):
            if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
                pairs.append((turns[i], turns[i + 1]))
        
        # 슬라이딩 윈도우로 샘플 생성
        if len(pairs) <= max_turns:
            expanded.append(sample)
        else:
            for start in range(len(pairs) - max_turns + 1):
                window = pairs[start:start + max_turns]
                new_messages = [system]
                for user, assistant in window:
                    new_messages.extend([user, assistant])
                expanded.append({"messages": new_messages})
    
    return expanded


def save_jsonl(data: list[dict], file_path: str):
    """JSONL 파일 저장"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ 저장됨: {file_path} ({len(data)}개 샘플)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="상담 대화 데이터셋 준비")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/conversations",
        help="입력 대화 파일 디렉토리"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetuning/data",
        help="출력 JSONL 디렉토리"
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=3,
        help="샘플당 최대 대화 턴 수"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="학습 데이터 비율"
    )
    
    args = parser.parse_args()
    
    # 모든 대화 파일 처리
    input_path = Path(args.input_dir)
    all_conversations = []
    
    for file in input_path.glob("*.txt"):
        print(f"📄 처리 중: {file.name}")
        convs = parse_conversation_file(str(file))
        all_conversations.extend(convs)
        print(f"   → {len(convs)}개 대화 추출")
    
    if not all_conversations:
        print("⚠️ 대화 데이터를 찾을 수 없습니다.")
        return
    
    # 형식 변환
    dataset = convert_to_chat_format(all_conversations)
    print(f"\n📊 변환된 샘플: {len(dataset)}개")
    
    # 멀티턴 분할
    expanded = split_multi_turn_conversations(dataset, args.max_turns)
    print(f"📊 분할 후 샘플: {len(expanded)}개")
    
    # 학습/검증 분할
    import random
    random.seed(42)
    random.shuffle(expanded)
    
    split_idx = int(len(expanded) * args.train_ratio)
    train_data = expanded[:split_idx]
    val_data = expanded[split_idx:]
    
    # 저장
    output_path = Path(args.output_dir)
    save_jsonl(train_data, output_path / "train_counseling.jsonl")
    save_jsonl(val_data, output_path / "val_counseling.jsonl")
    
    # 샘플 출력
    if train_data:
        print("\n📝 샘플 데이터:")
        sample = train_data[0]
        for msg in sample["messages"][:5]:
            role = msg["role"]
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            print(f"  [{role}] {content}")


if __name__ == "__main__":
    main()
