"""
통합 대화 데이터셋 전처리 스크립트 (v2)
- conversations/*.txt → 파인튜닝용 JSONL
- 다양한 양식 통합 처리 (NFD 파일명 포함)
- 데이터 증강: 슬라이딩 윈도우, 동일 주제 합성 멀티턴
- 출력: train_counseling.jsonl / val_counseling.jsonl (Kanana 학습용)

사용법:
    python prepare_finetuning_data.py
    python prepare_finetuning_data.py --augment --max_turns 4
"""

import os
import re
import json
import random
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# ==========================================
# 설정
# ==========================================
CONVERSATIONS_DIR = Path(__file__).parent.parent / "data" / "conversations"
OUTPUT_DIR = Path(__file__).parent / "data"
TRAIN_RATIO = 0.85

# 역할 매핑
USER_ROLES = ["고령자", "어르신", "이용자", "user", "user "]
ASSISTANT_ROLES = ["상담사", "관리사", "건강관리사", "agent", "assistant"]

# 시스템 프롬프트 (학습에 사용될 통일된 프롬프트)
SYSTEM_PROMPT = "너는 노인건강전문상담사로서 어르신의 건강 고민에 공감하며 일상에서 실천할 수 있는 건강 습관을 알려주고, 증상이 심각한 경우 의사 진료를 권유한다."


# ==========================================
# 텍스트 정리
# ==========================================

def clean_text(text: str) -> str:
    """텍스트 정리"""
    if not text:
        return ""
    text = re.sub(r'[ᄀ-ᅟᅠ-ᆿㄱ-ㅎㅏ-ㅣ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\[검색[^\]]*\]', '', text)
    text = re.sub(r'\[데이터베이스검색\]', '', text)
    text = re.sub(r'\[검색해서[^\]]*\]', '', text)
    text = re.sub(r'\[응답에 따라서[^\]]*\]', '', text)
    text = re.sub(r'\[약 종류[^\]]*\]', '', text)
    text = re.sub(r'\?\?+', '', text)
    return text.strip()


def normalize_role(role: str) -> str:
    """역할 표준화"""
    role_clean = role.strip().lower()
    # 상담사(적극적 시술권유) 같은 변형 처리
    role_clean = re.sub(r'\([^)]*\)', '', role_clean).strip()

    for r in USER_ROLES:
        if r.lower() in role_clean:
            return "user"
    for r in ASSISTANT_ROLES:
        if r.lower() in role_clean:
            return "assistant"
    if "system" in role_clean:
        return "system"
    return role_clean


# ==========================================
# 파싱 로직
# ==========================================

def extract_turns(text: str) -> List[Dict]:
    """텍스트에서 대화 턴 추출 (모든 양식 통합)"""
    turns = []

    # 역할 패턴 (불릿 •, 다양한 역할명, 괄호 태도 표시 포함)
    role_pattern = (
        r'(?:^|\n)[•·]?\s*'
        r'(System|system|고령자|어르신|이용자|상담사|관리사|건강관리사|'
        r'User|USer|user|Agent|agent|Assistant|assistant)'
        r'\s*(?:\([^)]*\))?'  # 선택적 괄호 (페르소나 정의), (적극적 시술권유) 등
        r'\s*[:：]\s*'
    )

    parts = re.split(role_pattern, text, flags=re.MULTILINE | re.IGNORECASE)

    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            role = parts[i].strip()
            message = parts[i + 1].strip()
            # 다음 섹션 마커 전까지만
            message = re.split(r'(?=<[^>]+>|\[사례|\[상황)', message)[0].strip()
            message = clean_text(message)

            if message and len(message) > 2:
                normalized = normalize_role(role)
                if normalized in ("user", "assistant"):
                    turns.append({"role": normalized, "content": message})

    return turns


def parse_file(filepath: Path) -> List[Dict]:
    """
    대화 파일을 파싱하여 개별 대화 목록 반환
    각 대화: {"topic": str, "turns": [{"role", "content"}, ...]}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    filename = unicodedata.normalize('NFC', filepath.stem)
    conversations = []

    # === 섹션 분리 전략 ===
    # 1) <제목> 패턴
    # 2) [사례 N] / [상황 N] 패턴
    # 3) 사례 N: 패턴 (손발저림)
    # 4) System 프롬프트로 분리

    # <제목> 분리 시도
    angle_sections = re.split(r'(<[^>]+>)', content)
    if len(angle_sections) > 2:
        return _parse_angle_bracket_format(angle_sections, filename)

    # [사례/상황] 분리 시도
    bracket_match = re.split(r'(\[(?:사례|상황)\s*\d+[^\]]*\])', content)
    if len(bracket_match) > 2:
        return _parse_bracket_format(bracket_match, filename)

    # 사례 N: 분리 시도 (손발저림)
    case_match = re.split(r'(사례\s*\d+\s*[:：])', content)
    if len(case_match) > 2:
        return _parse_case_format(case_match, filename)

    # System 프롬프트 분리
    system_match = re.split(
        r'(System\s*(?:\([^)]*\))?\s*[:：])',
        content, flags=re.IGNORECASE
    )
    if len(system_match) > 2:
        return _parse_system_format(system_match, filename)

    # 폴백: 전체를 하나의 대화로
    turns = extract_turns(content)
    if turns:
        cleaned = _clean_turns(turns)
        if cleaned:
            conversations.append({"topic": filename, "turns": cleaned})

    return conversations


def _parse_angle_bracket_format(sections, filename) -> List[Dict]:
    """<제목> 양식 파싱"""
    conversations = []
    current_topic = filename
    current_text = ""

    for part in sections:
        part = part.strip()
        if not part:
            continue

        if re.match(r'<[^>]+>', part):
            # 이전 섹션 처리
            if current_text:
                turns = extract_turns(current_text)
                cleaned = _clean_turns(turns)
                if cleaned:
                    conversations.append({"topic": current_topic, "turns": cleaned})
            current_topic = re.sub(r'[<>]', '', part).strip()
            current_text = ""
        else:
            current_text += " " + part

    # 마지막 섹션
    if current_text:
        turns = extract_turns(current_text)
        cleaned = _clean_turns(turns)
        if cleaned:
            conversations.append({"topic": current_topic, "turns": cleaned})

    return conversations


def _parse_bracket_format(sections, filename) -> List[Dict]:
    """[사례 N] / [상황 N] 양식 파싱"""
    conversations = []
    current_topic = filename
    current_text = ""

    for part in sections:
        part = part.strip()
        if not part:
            continue

        if re.match(r'\[(?:사례|상황)', part):
            if current_text:
                turns = extract_turns(current_text)
                cleaned = _clean_turns(turns)
                if cleaned:
                    conversations.append({"topic": current_topic, "turns": cleaned})
            current_topic = re.sub(r'[\[\]]', '', part).strip()
            current_text = ""
        else:
            current_text += " " + part

    if current_text:
        turns = extract_turns(current_text)
        cleaned = _clean_turns(turns)
        if cleaned:
            conversations.append({"topic": current_topic, "turns": cleaned})

    return conversations


def _parse_case_format(sections, filename) -> List[Dict]:
    """사례 N: 양식 파싱 (불릿 형식)"""
    conversations = []
    current_topic = filename
    current_text = ""

    for part in sections:
        part = part.strip()
        if not part:
            continue

        if re.match(r'사례\s*\d+', part):
            if current_text:
                turns = extract_turns(current_text)
                cleaned = _clean_turns(turns)
                if cleaned:
                    conversations.append({"topic": current_topic, "turns": cleaned})
            current_topic = part.rstrip(':：').strip()
            current_text = ""
        else:
            current_text += " " + part

    if current_text:
        turns = extract_turns(current_text)
        cleaned = _clean_turns(turns)
        if cleaned:
            conversations.append({"topic": current_topic, "turns": cleaned})

    return conversations


def _parse_system_format(sections, filename) -> List[Dict]:
    """System 프롬프트로 분리하는 양식"""
    conversations = []
    current_text = ""

    for part in sections:
        part = part.strip()
        if not part:
            continue

        if re.match(r'System', part, re.IGNORECASE):
            if current_text:
                turns = extract_turns(current_text)
                cleaned = _clean_turns(turns)
                if cleaned:
                    conversations.append({"topic": filename, "turns": cleaned})
            current_text = ""
        else:
            current_text += " " + part

    if current_text:
        turns = extract_turns(current_text)
        cleaned = _clean_turns(turns)
        if cleaned:
            conversations.append({"topic": filename, "turns": cleaned})

    return conversations


def _clean_turns(turns: List[Dict]) -> List[Dict]:
    """대화 턴 정리: 연속 동일 역할 병합, user 시작 보장"""
    if not turns:
        return []

    cleaned = []
    prev_role = None

    for turn in turns:
        role = turn["role"]
        content = turn["content"]

        if not content or len(content) < 3:
            continue

        if role == prev_role and cleaned:
            cleaned[-1]["content"] += " " + content
        else:
            cleaned.append({"role": role, "content": content})
            prev_role = role

    # user로 시작하도록
    while cleaned and cleaned[0]["role"] != "user":
        cleaned.pop(0)

    # 최소 user + assistant 1쌍
    if len(cleaned) < 2:
        return []

    has_user = any(t["role"] == "user" for t in cleaned)
    has_asst = any(t["role"] == "assistant" for t in cleaned)
    if not (has_user and has_asst):
        return []

    return cleaned


# ==========================================
# 데이터 증강
# ==========================================

def augment_sliding_window(conversations: List[Dict], window_size: int = 4) -> List[Dict]:
    """
    멀티턴 대화를 슬라이딩 윈도우로 증강
    - 원본 유지 + 서브 시퀀스 생성
    """
    augmented = []

    for conv in conversations:
        turns = conv["turns"]
        topic = conv.get("topic", "")

        # 원본 항상 포함
        augmented.append(conv)

        # user-assistant 쌍 추출
        pairs = []
        for i in range(0, len(turns) - 1, 2):
            if turns[i]["role"] == "user" and i + 1 < len(turns) and turns[i + 1]["role"] == "assistant":
                pairs.append((turns[i], turns[i + 1]))

        # 3쌍 이상이면 슬라이딩 윈도우
        if len(pairs) >= 3:
            for start in range(1, len(pairs) - 1):
                end = min(start + window_size, len(pairs))
                if end - start >= 2:
                    sub_turns = []
                    for u, a in pairs[start:end]:
                        sub_turns.extend([u, a])
                    augmented.append({"topic": topic, "turns": sub_turns})

    return augmented


def augment_combine_singles(conversations: List[Dict], max_combine: int = 3) -> List[Dict]:
    """
    같은 주제(파일)의 단일턴 대화들을 모아 합성 멀티턴 대화 생성
    """
    # 파일명(주제) 기준으로 단일턴 대화 그룹화
    topic_singles = defaultdict(list)

    for conv in conversations:
        turns = conv["turns"]
        if len(turns) == 2:  # user + assistant 1쌍
            # 파일명에서 주제 추출
            topic_key = conv.get("topic", "").split("_")[0].split(" ")[0]
            topic_singles[topic_key].append(conv)

    combined = []
    for topic, singles in topic_singles.items():
        if len(singles) < 2:
            continue

        # 2~max_combine개씩 묶어서 합성 멀티턴 생성
        random.shuffle(singles)
        for i in range(0, len(singles) - 1, max_combine):
            group = singles[i:i + max_combine]
            if len(group) >= 2:
                merged_turns = []
                for conv in group:
                    merged_turns.extend(conv["turns"])
                combined.append({"topic": topic + "_합성", "turns": merged_turns})

    return combined


# ==========================================
# 출력 변환
# ==========================================

def to_chat_format(conv: Dict) -> Dict:
    """ChatML 형식으로 변환 (system + user/assistant 턴)"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in conv.get("turns", []):
        content = clean_text(turn["content"])
        if content and len(content) > 2:
            messages.append({"role": turn["role"], "content": content})

    return {"messages": messages}


def validate(conv: Dict) -> bool:
    """유효성 검증"""
    msgs = conv.get("messages", [])
    if len(msgs) < 3:  # system + user + assistant
        return False
    has_user = any(m["role"] == "user" for m in msgs)
    has_asst = any(m["role"] == "assistant" for m in msgs)
    return has_user and has_asst


# ==========================================
# 메인
# ==========================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="파인튜닝 데이터 전처리 (v2)")
    parser.add_argument("--augment", action="store_true", help="데이터 증강 활성화")
    parser.add_argument("--max_turns", type=int, default=4, help="슬라이딩 윈도우 크기")
    parser.add_argument("--combine", type=int, default=3, help="단일턴 합성 시 최대 묶음 수")
    args = parser.parse_args()

    print("=" * 60)
    print("📊 파인튜닝 데이터 전처리 v2")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_conversations = []
    file_stats = {}

    # 모든 대화 파일 처리
    for filepath in sorted(CONVERSATIONS_DIR.iterdir()):
        if not filepath.suffix == '.txt':
            continue

        display_name = unicodedata.normalize('NFC', filepath.name)
        print(f"\n📄 {display_name}")

        try:
            convs = parse_file(filepath)
            turn_counts = [len(c["turns"]) for c in convs]
            multi = sum(1 for t in turn_counts if t > 2)
            single = sum(1 for t in turn_counts if t == 2)

            print(f"   파싱: {len(convs)}개 (멀티턴: {multi}, 싱글턴: {single})")

            file_stats[display_name] = {
                "total": len(convs),
                "multi": multi,
                "single": single
            }

            all_conversations.extend(convs)

        except Exception as e:
            print(f"   ❌ 오류: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"총 파싱된 대화: {len(all_conversations)}개")

    # 데이터 증강
    if args.augment:
        print(f"\n🔄 데이터 증강 중...")
        before = len(all_conversations)

        # 슬라이딩 윈도우
        augmented = augment_sliding_window(all_conversations, args.max_turns)
        print(f"   슬라이딩 윈도우: {before} → {len(augmented)}")

        # 단일턴 합성
        combined = augment_combine_singles(all_conversations, args.combine)
        augmented.extend(combined)
        print(f"   단일턴 합성: +{len(combined)}개")

        all_conversations = augmented
        print(f"   증강 후 총: {len(all_conversations)}개")

    # ChatML 형식 변환 + 유효성 검증
    dataset = []
    for conv in all_conversations:
        chat = to_chat_format(conv)
        if validate(chat):
            dataset.append(chat)

    print(f"유효한 학습 데이터: {len(dataset)}개")

    # 통계
    turn_dist = defaultdict(int)
    total_tokens = 0
    for d in dataset:
        non_sys = [m for m in d["messages"] if m["role"] != "system"]
        turn_dist[len(non_sys)] += 1
        total_tokens += sum(len(m["content"]) * 2.5 for m in d["messages"])

    print(f"\n📊 턴 수 분포:")
    for t in sorted(turn_dist.keys()):
        print(f"   {t}턴: {turn_dist[t]}개")
    print(f"   대략적 총 토큰: {int(total_tokens):,}")

    # 셔플 + 분할
    random.seed(42)
    random.shuffle(dataset)

    split_idx = int(len(dataset) * TRAIN_RATIO)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    print(f"\n📊 데이터셋 분할:")
    print(f"   Train: {len(train_data)}개")
    print(f"   Valid: {len(val_data)}개")

    # 저장 (train_kanana_lora.py가 읽는 경로와 통일)
    train_path = OUTPUT_DIR / "train_counseling.jsonl"
    val_path = OUTPUT_DIR / "val_counseling.jsonl"

    for path, data in [(train_path, train_data), (val_path, val_data)]:
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # healthcare_conversations 경로에도 저장 (호환성)
    hc_train_path = OUTPUT_DIR / "healthcare_conversations_train.jsonl"
    hc_val_path = OUTPUT_DIR / "healthcare_conversations_valid.jsonl"
    for path, data in [(hc_train_path, train_data), (hc_val_path, val_data)]:
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n✅ 저장 완료:")
    print(f"   {train_path}")
    print(f"   {val_path}")

    # 샘플 출력
    print(f"\n{'='*60}")
    print("📝 샘플 데이터 (처음 3개):")
    print("="*60)
    for idx, sample in enumerate(train_data[:3]):
        print(f"\n--- 샘플 {idx+1} (턴: {len(sample['messages'])-1}) ---")
        for msg in sample["messages"]:
            role = msg["role"].upper()
            content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
            print(f"  [{role}] {content}")

    # 파일별 통계
    print(f"\n{'='*60}")
    print("파일별 통계:")
    print("="*60)
    for fname, stats in sorted(file_stats.items()):
        print(f"  {fname}: {stats['total']}개 (멀티:{stats['multi']}, 싱글:{stats['single']})")


if __name__ == "__main__":
    main()
