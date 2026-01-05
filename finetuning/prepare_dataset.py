"""
데이터셋 전처리 스크립트
치매노인-생활지원사 대화 데이터를 파인튜닝 형식으로 변환
"""

import json
import os
from pathlib import Path
from typing import Optional
import argparse
from dataclasses import dataclass


@dataclass
class ConversationTurn:
    """대화 턴"""
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ConversationSample:
    """대화 샘플"""
    conversation_id: str
    patient_info: Optional[str]
    turns: list[ConversationTurn]


def load_jsonl(file_path: str) -> list[dict]:
    """JSONL 파일 로드"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], file_path: str):
    """JSONL 파일 저장"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_to_chat_format(conversations: list[dict]) -> list[dict]:
    """
    원본 대화 데이터를 채팅 형식으로 변환
    
    입력 형식 (예시):
    {
        "id": "conv_001",
        "patient_info": "80세 여성, 경도 치매",
        "dialogue": [
            {"speaker": "patient", "text": "오늘 약 먹었나?"},
            {"speaker": "caregiver", "text": "네, 어르신. 아침에 드셨어요."}
        ]
    }
    
    출력 형식 (Qwen 학습용):
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "오늘 약 먹었나?"},
            {"role": "assistant", "content": "네, 어르신. 아침에 드셨어요."}
        ]
    }
    """
    system_prompt = """당신은 치매노인을 돌보는 따뜻하고 친절한 AI 도우미입니다. 
항상 존댓말을 사용하고, 천천히 명확하게 설명합니다.
복잡한 내용은 짧고 간단한 문장으로 나눠서 전달합니다.
환자의 감정을 존중하고 공감하며 대화합니다."""
    
    converted = []
    
    for conv in conversations:
        messages = [{"role": "system", "content": system_prompt}]
        
        # 환자 정보가 있으면 시스템 프롬프트에 추가
        if conv.get("patient_info"):
            messages[0]["content"] += f"\n\n환자 정보: {conv['patient_info']}"
        
        # 대화 턴 변환
        dialogue = conv.get("dialogue", conv.get("turns", []))
        for turn in dialogue:
            speaker = turn.get("speaker", turn.get("role", ""))
            text = turn.get("text", turn.get("content", ""))
            
            if speaker in ["patient", "user", "환자"]:
                messages.append({"role": "user", "content": text})
            elif speaker in ["caregiver", "assistant", "생활지원사", "AI"]:
                messages.append({"role": "assistant", "content": text})
        
        # user와 assistant가 번갈아 나오도록 정리
        if len(messages) > 1:
            converted.append({"messages": messages})
    
    return converted


def convert_to_instruction_format(conversations: list[dict]) -> list[dict]:
    """
    Instruction 형식으로 변환 (Alpaca 스타일)
    
    출력 형식:
    {
        "instruction": "치매노인의 질문에 친절하게 답변하세요.",
        "input": "오늘 약 먹었나?",
        "output": "네, 어르신. 아침에 드셨어요."
    }
    """
    converted = []
    base_instruction = "치매노인을 돌보는 AI 도우미로서, 다음 질문에 따뜻하고 친절하게 답변하세요. 항상 존댓말을 사용하고, 간단명료하게 설명합니다."
    
    for conv in conversations:
        dialogue = conv.get("dialogue", conv.get("turns", []))
        patient_info = conv.get("patient_info", "")
        
        # 연속된 user-assistant 쌍 추출
        i = 0
        while i < len(dialogue) - 1:
            current = dialogue[i]
            next_turn = dialogue[i + 1]
            
            current_speaker = current.get("speaker", current.get("role", ""))
            next_speaker = next_turn.get("speaker", next_turn.get("role", ""))
            
            if current_speaker in ["patient", "user", "환자"] and \
               next_speaker in ["caregiver", "assistant", "생활지원사", "AI"]:
                
                instruction = base_instruction
                if patient_info:
                    instruction += f"\n환자 정보: {patient_info}"
                
                converted.append({
                    "instruction": instruction,
                    "input": current.get("text", current.get("content", "")),
                    "output": next_turn.get("text", next_turn.get("content", ""))
                })
                i += 2
            else:
                i += 1
    
    return converted


def create_sample_data(output_dir: str):
    """샘플 데이터 생성"""
    sample_conversations = [
        {
            "id": "conv_001",
            "patient_info": "김영희, 82세 여성, 경도 치매, 고혈압 약 복용 중",
            "dialogue": [
                {"speaker": "patient", "text": "오늘 약 먹었나?"},
                {"speaker": "caregiver", "text": "네, 어르신. 오늘 아침 8시에 혈압약 드셨어요. 잘 드셨습니다."},
                {"speaker": "patient", "text": "아, 그랬구나. 밥은 먹었어?"},
                {"speaker": "caregiver", "text": "네, 아침 식사도 맛있게 드셨어요. 미역국이랑 밥 드셨습니다."},
            ]
        },
        {
            "id": "conv_002",
            "patient_info": "박철수, 78세 남성, 중등도 치매",
            "dialogue": [
                {"speaker": "patient", "text": "내 아들 어디 갔어?"},
                {"speaker": "caregiver", "text": "철수 어르신, 아드님은 회사에 출근하셨어요. 저녁에 오신다고 하셨습니다."},
                {"speaker": "patient", "text": "그래? 언제 와?"},
                {"speaker": "caregiver", "text": "6시쯤 오실 거예요. 조금만 기다리시면 곧 오실 거예요."},
            ]
        },
        {
            "id": "conv_003",
            "patient_info": "이순자, 85세 여성, 경도 치매, 당뇨 관리 중",
            "dialogue": [
                {"speaker": "patient", "text": "머리가 좀 아파."},
                {"speaker": "caregiver", "text": "어르신, 머리가 아프시군요. 많이 아프세요? 좀 쉬시면서 물 한 잔 드실까요?"},
                {"speaker": "patient", "text": "응, 좀 아파. 물 줘."},
                {"speaker": "caregiver", "text": "네, 여기 물이요. 천천히 드세요. 계속 아프시면 말씀해 주세요. 필요하면 보호자분께 연락드릴게요."},
            ]
        },
        {
            "id": "conv_004",
            "patient_info": "최동수, 80세 남성, 중등도 치매",
            "dialogue": [
                {"speaker": "patient", "text": "산책 가고 싶어."},
                {"speaker": "caregiver", "text": "좋은 생각이에요, 어르신! 오늘 날씨가 좋아서 산책하기 딱 좋아요. 겉옷 입고 나가실까요?"},
                {"speaker": "patient", "text": "그래, 나가자."},
                {"speaker": "caregiver", "text": "네, 모자도 쓰시고 천천히 나가요. 오늘은 공원까지 걸어볼까요?"},
            ]
        },
        {
            "id": "conv_005",
            "patient_info": "정미경, 79세 여성, 경도 치매",
            "dialogue": [
                {"speaker": "patient", "text": "여기가 어디야?"},
                {"speaker": "caregiver", "text": "어르신, 여기는 어르신 댁이에요. 미경 어르신 집이요. 거실에 계세요."},
                {"speaker": "patient", "text": "우리 집이야?"},
                {"speaker": "caregiver", "text": "네, 맞아요. 어르신이 30년 넘게 사신 집이에요. 저기 가족사진도 있잖아요."},
            ]
        }
    ]
    
    # 원본 데이터 저장
    os.makedirs(output_dir, exist_ok=True)
    save_jsonl(sample_conversations, os.path.join(output_dir, "raw_conversations.jsonl"))
    
    # 학습 형식으로 변환
    chat_format = convert_to_chat_format(sample_conversations)
    save_jsonl(chat_format, os.path.join(output_dir, "train_chat.jsonl"))
    
    instruction_format = convert_to_instruction_format(sample_conversations)
    save_jsonl(instruction_format, os.path.join(output_dir, "train_instruction.jsonl"))
    
    print(f"✅ 샘플 데이터 생성 완료:")
    print(f"   - 원본: {output_dir}/raw_conversations.jsonl ({len(sample_conversations)}개)")
    print(f"   - 채팅 형식: {output_dir}/train_chat.jsonl ({len(chat_format)}개)")
    print(f"   - 인스트럭션 형식: {output_dir}/train_instruction.jsonl ({len(instruction_format)}개)")


def prepare_dataset(
    input_path: str,
    output_dir: str,
    format_type: str = "chat",
    train_ratio: float = 0.9
):
    """
    데이터셋 준비
    
    Args:
        input_path: 입력 파일 경로
        output_dir: 출력 디렉토리
        format_type: 출력 형식 ('chat' 또는 'instruction')
        train_ratio: 학습 데이터 비율
    """
    print(f"📂 데이터 로드 중: {input_path}")
    conversations = load_jsonl(input_path)
    print(f"   총 {len(conversations)}개 대화 로드됨")
    
    # 형식 변환
    if format_type == "chat":
        converted = convert_to_chat_format(conversations)
    else:
        converted = convert_to_instruction_format(conversations)
    
    print(f"   {len(converted)}개 샘플로 변환됨")
    
    # Train/Val 분할
    import random
    random.shuffle(converted)
    
    split_idx = int(len(converted) * train_ratio)
    train_data = converted[:split_idx]
    val_data = converted[split_idx:]
    
    # 저장
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, f"train_{format_type}.jsonl")
    val_path = os.path.join(output_dir, f"val_{format_type}.jsonl")
    
    save_jsonl(train_data, train_path)
    save_jsonl(val_data, val_path)
    
    print(f"✅ 데이터셋 준비 완료:")
    print(f"   - 학습: {train_path} ({len(train_data)}개)")
    print(f"   - 검증: {val_path} ({len(val_data)}개)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="데이터셋 전처리")
    parser.add_argument("--input", type=str, help="입력 파일 경로")
    parser.add_argument("--output", type=str, default="./data/conversations", help="출력 디렉토리")
    parser.add_argument("--format", type=str, default="chat", choices=["chat", "instruction"], help="출력 형식")
    parser.add_argument("--create-sample", action="store_true", help="샘플 데이터 생성")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data(args.output)
    elif args.input:
        prepare_dataset(args.input, args.output, args.format)
    else:
        print("--input 또는 --create-sample 옵션을 지정하세요.")
        print("예: python prepare_dataset.py --create-sample")
        print("예: python prepare_dataset.py --input data.jsonl --output ./data/conversations")
