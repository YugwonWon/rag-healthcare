"""
대화 예제 파일들을 파싱하여 train/valid 데이터셋으로 변환
다양한 형식의 파일들을 통합 처리
- 각 사례/상황별로 개별 대화로 분리
- 멀티턴 대화 지원
"""

import os
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

# 설정
CONVERSATIONS_DIR = Path(__file__).parent.parent / "data" / "conversations"
OUTPUT_DIR = Path(__file__).parent / "data"
TRAIN_RATIO = 0.85  # 85% train, 15% valid

# 역할 매핑 (다양한 표현을 표준화)
USER_ROLES = ["고령자", "어르신", "이용자", "user"]
ASSISTANT_ROLES = ["상담사", "관리사", "건강관리사", "agent", "assistant"]

# 기본 System 프롬프트
DEFAULT_SYSTEM = "너는 노인건강전문상담사로서 어르신의 건강 고민에 공감하며 일상에서 실천할 수 있는 건강 습관을 알려주고, 증상이 심각한 경우 의사 진료를 권유한다."


def clean_text(text: str) -> str:
    """텍스트 정리: 깨진 문자, 불필요한 공백 제거"""
    if not text:
        return ""
    # 깨진 한글 문자 제거 (자모음 분리된 문자)
    text = re.sub(r'[ᄀ-ᅟᅠ-ᆿㄱ-ㅎㅏ-ㅣ]', '', text)
    # 불필요한 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()
    # [검색], [데이터베이스검색] 등 플레이스홀더 제거
    text = re.sub(r'\[검색[^\]]*\]', '', text)
    text = re.sub(r'\[데이터베이스검색\]', '', text)
    # ?? 같은 불완전한 마커 제거
    text = re.sub(r'\?\?+', '', text)
    return text.strip()


def normalize_role(role: str) -> str:
    """역할을 표준 형식으로 변환"""
    role_clean = role.strip().lower()
    
    for r in USER_ROLES:
        if r.lower() in role_clean:
            return "user"
    
    for r in ASSISTANT_ROLES:
        if r.lower() in role_clean:
            return "assistant"
    
    if "system" in role_clean:
        return "system"
    
    return role_clean


def split_by_section(content: str) -> List[str]:
    """
    다양한 섹션 구분자로 콘텐츠 분리
    - <제목> 형태
    - [사례 N], [상황 N]
    - 사례 N:
    - System 프롬프트 시작점
    """
    # 섹션 구분자 패턴들
    section_patterns = [
        r'<[^>]+>',  # <갑자기 얼굴이 확 달아오르고 땀이 남>
        r'\[사례\s*\d+\]',  # [사례 1]
        r'\[상황\s*\d+[^\]]*\]',  # [상황 1: ...]
        r'(?:^|\n)사례\s*\d+\s*[:：]',  # 사례 1:
        r'(?:^|\n)System\s*(?:\([^)]*\))?\s*[:：]',  # System:
    ]
    
    combined_pattern = '|'.join(f'({p})' for p in section_patterns)
    
    # 섹션 분리
    parts = re.split(combined_pattern, content, flags=re.MULTILINE | re.IGNORECASE)
    
    sections = []
    current_section = ""
    
    for part in parts:
        if part is None:
            continue
        part = part.strip()
        if not part:
            continue
            
        # 섹션 구분자인 경우
        is_separator = any(re.match(p, part, re.IGNORECASE) for p in section_patterns)
        
        if is_separator:
            if current_section.strip():
                sections.append(current_section.strip())
            current_section = part
        else:
            current_section += " " + part
    
    if current_section.strip():
        sections.append(current_section.strip())
    
    return sections if sections else [content]


def extract_turns_from_text(text: str) -> List[Dict]:
    """
    텍스트에서 대화 턴 추출
    역할: 메시지 형태 또는 • 역할: 메시지 형태 모두 처리
    """
    turns = []
    
    # 역할 패턴: "역할:" 또는 "• 역할:"
    role_pattern = r'(?:^|\n)[•·]?\s*(System|system|고령자|어르신|이용자|상담사|관리사|건강관리사|User|USer|user|Agent|agent|Assistant|assistant)\s*(?:\([^)]*\))?\s*[:：]\s*'
    
    # 역할로 텍스트 분리
    parts = re.split(role_pattern, text, flags=re.MULTILINE | re.IGNORECASE)
    
    # parts: [prefix, role1, msg1, role2, msg2, ...]
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            role = parts[i].strip()
            message = parts[i + 1].strip()
            
            # 메시지 정리
            message = re.sub(r'\s+', ' ', message).strip()
            # 다음 섹션 마커 전까지만
            message = re.split(r'(?=<[^>]+>|\[사례|\[상황)', message)[0].strip()
            
            if message and len(message) > 2:
                normalized_role = normalize_role(role)
                turns.append({
                    "role": normalized_role,
                    "content": message
                })
    
    return turns


def parse_conversation_file(filepath: Path) -> List[Dict]:
    """
    대화 파일을 파싱하여 개별 대화 목록 반환
    각 대화는 {"system": str, "turns": [{"role": str, "content": str}, ...]} 형태
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    filename = filepath.name
    conversations = []
    
    # ca_sample.txt 특별 처리 (상황 형식)
    if filename == "ca_sample.txt":
        situations = re.split(r'\[상황\s*\d+[^\]]*\]', content)
        for situation in situations:
            if not situation.strip():
                continue
            turns = extract_turns_from_text(situation)
            if turns:
                # 연속된 동일 역할 제거, user/assistant 번갈아 나오도록
                cleaned_turns = clean_consecutive_roles(turns)
                if cleaned_turns:
                    conversations.append({
                        "system": DEFAULT_SYSTEM,
                        "turns": cleaned_turns
                    })
        return conversations
    
    # 손발저림 대화예제.txt 특별 처리 (사례: 불릿 형식)
    if "손발저림" in filename:
        cases = re.split(r'사례\s*\d+\s*[:：]', content)
        for case in cases:
            if not case.strip():
                continue
            turns = extract_turns_from_text(case)
            if turns:
                cleaned_turns = clean_consecutive_roles(turns)
                if cleaned_turns:
                    conversations.append({
                        "system": DEFAULT_SYSTEM,
                        "turns": cleaned_turns
                    })
        return conversations
    
    # 일반 형식: System 프롬프트 또는 <제목> 섹션으로 분리
    # 먼저 <제목> 패턴으로 분리 시도
    sections = re.split(r'(<[^>]+>)', content)
    
    current_system = ""
    current_turns = []
    
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        
        # <제목> 마커인 경우 (새 대화 시작)
        if re.match(r'<[^>]+>', section):
            # 이전 대화 저장
            if current_turns:
                cleaned = clean_consecutive_roles(current_turns)
                if cleaned:
                    conversations.append({
                        "system": current_system or DEFAULT_SYSTEM,
                        "turns": cleaned
                    })
            current_system = ""
            current_turns = []
            continue
        
        # System 프롬프트 패턴으로 다시 분리
        system_parts = re.split(r'(System\s*(?:\([^)]*\))?\s*[:：])', section, flags=re.IGNORECASE)
        
        for j, part in enumerate(system_parts):
            part = part.strip()
            if not part:
                continue
            
            # System 마커인 경우
            if re.match(r'System', part, re.IGNORECASE):
                # 이전 대화 저장
                if current_turns:
                    cleaned = clean_consecutive_roles(current_turns)
                    if cleaned:
                        conversations.append({
                            "system": current_system or DEFAULT_SYSTEM,
                            "turns": cleaned
                        })
                current_turns = []
                continue
            
            # System 프롬프트 추출
            system_match = re.match(r'(.+?)(?=고령자|어르신|이용자|User|USer)', part, re.DOTALL | re.IGNORECASE)
            if system_match:
                current_system = system_match.group(1).strip()
                # System 이후 텍스트에서 턴 추출
                remaining = part[system_match.end()-len(system_match.group(0).split()[-1]):]
                turns = extract_turns_from_text(part)
            else:
                turns = extract_turns_from_text(part)
            
            current_turns.extend(turns)
    
    # 마지막 대화 저장
    if current_turns:
        cleaned = clean_consecutive_roles(current_turns)
        if cleaned:
            conversations.append({
                "system": current_system or DEFAULT_SYSTEM,
                "turns": cleaned
            })
    
    return conversations


def clean_consecutive_roles(turns: List[Dict]) -> List[Dict]:
    """
    연속된 동일 역할 메시지 병합 및 정리
    system 역할 제거 (별도로 처리)
    """
    if not turns:
        return []
    
    # system 역할 제거
    turns = [t for t in turns if t["role"] in ["user", "assistant"]]
    
    if not turns:
        return []
    
    cleaned = []
    prev_role = None
    
    for turn in turns:
        role = turn["role"]
        content = clean_text(turn["content"])
        
        # 빈 메시지나 너무 짧은 메시지 건너뛰기
        if not content or len(content) < 3:
            continue
        
        # 연속된 동일 역할이면 병합
        if role == prev_role and cleaned:
            cleaned[-1]["content"] += " " + content
        else:
            cleaned.append({"role": role, "content": content})
            prev_role = role
    
    # 첫 턴이 user가 아니면 제거
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)
    
    # 최소 1턴 이상 (user -> assistant)
    if len(cleaned) < 2:
        return []
    
    # user와 assistant 둘 다 있는지 확인
    has_user = any(t["role"] == "user" for t in cleaned)
    has_assistant = any(t["role"] == "assistant" for t in cleaned)
    
    if not (has_user and has_assistant):
        return []
    
    return cleaned


def parse_file(filepath: Path) -> List[Dict]:
    """파일 형식에 따라 적절한 파서 선택"""
    return parse_conversation_file(filepath)


def create_chat_format(conv: Dict) -> Dict:
    """
    대화를 ChatML 형식으로 변환
    """
    messages = []
    
    # System 메시지는 항상 기본 프롬프트 사용 (일관성을 위해)
    messages.append({"role": "system", "content": DEFAULT_SYSTEM})
    
    # 대화 턴
    for turn in conv.get("turns", []):
        content = clean_text(turn["content"])
        if content and len(content) > 2:
            messages.append({
                "role": turn["role"],
                "content": content
            })
    
    return {"messages": messages}


def validate_conversation(conv: Dict) -> bool:
    """대화 유효성 검증"""
    messages = conv.get("messages", [])
    
    # 최소 조건: system + user + assistant = 3개 이상
    if len(messages) < 3:
        return False
    
    # user와 assistant 각각 1개 이상
    has_user = any(m["role"] == "user" for m in messages)
    has_assistant = any(m["role"] == "assistant" for m in messages)
    
    return has_user and has_assistant


def split_multi_turn_conversations(conversations: List[Dict], max_turns: int = 6) -> List[Dict]:
    """
    긴 멀티턴 대화를 여러 개의 짧은 대화로 분리
    파인튜닝 데이터 증강을 위해
    """
    result = []
    
    for conv in conversations:
        system = conv.get("system", DEFAULT_SYSTEM)
        turns = conv.get("turns", [])
        
        if len(turns) <= max_turns:
            result.append(conv)
        else:
            # 긴 대화를 여러 개로 분리
            # 겹치는 컨텍스트를 유지하면서 분리
            for i in range(0, len(turns), max_turns - 2):
                chunk = turns[i:i + max_turns]
                
                # 첫 턴이 user인지 확인
                if chunk and chunk[0]["role"] == "user":
                    result.append({
                        "system": system,
                        "turns": chunk
                    })
    
    return result


def main():
    print("=" * 60)
    print("대화 예제 데이터셋 생성")
    print("=" * 60)
    
    # 출력 디렉토리 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_conversations = []
    file_stats = {}
    
    # 모든 파일 처리
    for filepath in sorted(CONVERSATIONS_DIR.glob("*.txt")):
        print(f"\n📄 처리 중: {filepath.name}")
        
        try:
            convs = parse_file(filepath)
            print(f"   파싱된 대화: {len(convs)}개")
            
            # 긴 대화 분리
            convs = split_multi_turn_conversations(convs, max_turns=8)
            print(f"   분리 후: {len(convs)}개")
            
            # ChatML 형식으로 변환
            chat_convs = []
            for conv in convs:
                chat_conv = create_chat_format(conv)
                if validate_conversation(chat_conv):
                    chat_convs.append(chat_conv)
            
            print(f"   유효한 대화: {len(chat_convs)}개")
            
            file_stats[filepath.name] = {
                "parsed": len(convs),
                "valid": len(chat_convs)
            }
            
            all_conversations.extend(chat_convs)
            
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n총 유효한 대화: {len(all_conversations)}개")
    
    # 셔플 후 train/valid 분리
    random.seed(42)
    random.shuffle(all_conversations)
    
    split_idx = int(len(all_conversations) * TRAIN_RATIO)
    train_data = all_conversations[:split_idx]
    valid_data = all_conversations[split_idx:]
    
    print(f"\n📊 데이터셋 분할:")
    print(f"   Train: {len(train_data)}개")
    print(f"   Valid: {len(valid_data)}개")
    
    # JSONL 형식으로 저장
    train_path = OUTPUT_DIR / "healthcare_conversations_train.jsonl"
    valid_path = OUTPUT_DIR / "healthcare_conversations_valid.jsonl"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for conv in train_data:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')
    
    with open(valid_path, 'w', encoding='utf-8') as f:
        for conv in valid_data:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 저장 완료:")
    print(f"   {train_path}")
    print(f"   {valid_path}")
    
    # 샘플 출력
    print("\n" + "=" * 60)
    print("샘플 데이터 (처음 3개):")
    print("=" * 60)
    
    for idx, sample in enumerate(train_data[:3]):
        print(f"\n--- 샘플 {idx + 1} ---")
        for msg in sample["messages"]:
            role = msg["role"].upper()
            content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
            print(f"[{role}] {content}")
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("파일별 통계:")
    print("=" * 60)
    total_valid = 0
    for filename, stats in sorted(file_stats.items()):
        print(f"  {filename}: {stats['parsed']}개 파싱 → {stats['valid']}개 유효")
        total_valid += stats['valid']
    
    print(f"\n  합계: {total_valid}개")


if __name__ == "__main__":
    main()
