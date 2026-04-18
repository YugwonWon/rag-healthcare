#!/usr/bin/env python3
"""
Base vs Fine-tuned 모델 반복 평가 (n=50)
=========================================
10개 프롬프트 × 5회 반복으로 통계적 신뢰도를 높인 모델 비교 평가.

사용법:
    cd /Users/yugwon/Projects/rag-healthcare
    python3 -m app.test.eval_model_repeated
"""

import json
import re
import time
import statistics
from pathlib import Path

import httpx

# ── 동일 프롬프트 ──
COMPARISON_PROMPTS = [
    {"id": "sleep_001", "intent": "health_consult",
     "query": "요즘 잠을 통 못 자겠어. 밤에 자다 깨고 또 깨고… 피곤해 죽겠어.", "topic": "수면장애"},
    {"id": "incontinence_001", "intent": "health_consult",
     "query": "재채기만 하면 소변이 새는데, 너무 부끄러워서 아무한테도 말 못 하겠어.", "topic": "요실금 (프라이버시 민감)"},
    {"id": "cognitive_001", "intent": "health_consult",
     "query": "딸이 자꾸 기억력이 떨어진다고 걱정하더라고.", "topic": "인지기능 (프라이버시 민감)"},
    {"id": "medication_001", "intent": "medication",
     "query": "혈압약을 며칠 안 먹었는데 괜찮을까?", "topic": "복약 관리"},
    {"id": "fall_001", "intent": "health_consult",
     "query": "어제 화장실 가다가 미끄러져서 넘어졌어. 무릎이 아파.", "topic": "낙상"},
    {"id": "emotion_001", "intent": "health_consult",
     "query": "요즘 뭘해도 재미가 없고 그냥 우울해.", "topic": "정서 건강"},
    {"id": "general_001", "intent": "general_chat",
     "query": "오늘 날씨가 좋으니까 기분이 좋아.", "topic": "일상 대화"},
    {"id": "emergency_001", "intent": "emergency",
     "query": "갑자기 가슴이 너무 아프고 숨이 안 쉬어져요.", "topic": "응급"},
    {"id": "lifestyle_001", "intent": "lifestyle",
     "query": "운동을 하고 싶은데 무릎이 안 좋아서 뭘 해야 할지 모르겠어.", "topic": "생활습관"},
    {"id": "privacy_001", "intent": "health_consult",
     "query": "아들이 나보고 치매 검사 받으라는데... 나는 괜찮은 것 같은데.", "topic": "인지기능 (가족+프라이버시)"},
]

OLLAMA_URL = "http://localhost:11434/api/generate"
BASE_MODEL = "kanana-base-raw"
FINETUNED_MODEL = "kanana-counseling"
NUM_TRIALS = 5


def auto_evaluate_response(query: str, resp: str, intent: str) -> dict:
    """단일 응답 품질 점수 산출 (기존 eval_roman.py 로직 동일)"""
    scores = {}

    # 1. 길이 적절성
    length = len(resp)
    if 20 <= length <= 300:
        scores["length"] = 1.0
    elif length < 20:
        scores["length"] = 0.3
    else:
        scores["length"] = 0.7

    # 2. 존댓말
    honorific_endings = ["요", "다", "세요", "니다", "습니다", "어요", "아요"]
    scores["honorific"] = 1.0 if any(resp.rstrip().endswith(e) for e in honorific_endings) else 0.5

    # 3. 공감 표현
    empathy_words = ["이해", "걱정", "힘드", "괴로", "불편", "안타깝", "많이", "그러시",
                     "아이go", "아이고", "속상", "고생", "마음"]
    scores["empathy"] = 1.0 if any(w in resp for w in empathy_words) else 0.0

    # 4. 목록/번호/마크다운 미사용
    has_list = bool(re.search(r'^\s*[\d\-\*·•][\.\)]\s', resp, re.MULTILINE))
    has_markdown = bool(re.search(r'\*\*[^*]+\*\*', resp))
    has_numbered = bool(re.search(r'^\s*\d+[\.\)]\s', resp, re.MULTILINE))
    scores["no_list"] = 0.0 if (has_list or has_markdown or has_numbered) else 1.0

    # 5. 금지 표현
    forbidden = ["도움이 되셨", "도움이 되었", "궁금한 점이 있으시면", "추가 질문"]
    scores["no_forbidden"] = 0.0 if any(f in resp for f in forbidden) else 1.0

    # 6. 응급 119
    if intent == "emergency":
        scores["emergency_119"] = 1.0 if "119" in resp else 0.0

    # 7. 프라이버시 보호
    invasive = ["치매입니까", "치매 증상", "요실금이 있으신가요", "우울증이 있으신가요"]
    scores["privacy"] = 0.0 if any(inv in resp for inv in invasive) else 1.0

    # 8. 구체적 건강 조언
    specific_advice = [
        "수영", "걷기", "산책", "스트레칭", "요가", "자전거",
        "수면", "카페인", "잠자리", "기상", "취침",
        "식이섬유", "물을", "채소", "과일", "영양", "식사",
        "혈압약", "복용", "정기적", "검진", "진료",
        "보조 기구", "지팡이", "미끄럼", "안전",
        "패드", "골반", "케겔",
        "병원", "전문의", "상담",
    ]
    advice_count = sum(1 for a in specific_advice if a in resp)
    scores["specific_advice"] = min(1.0, advice_count / 3.0)

    # 9. 자연스러운 마무리
    ends_natural = resp.rstrip().endswith("?") or resp.rstrip().endswith("요?") or \
                   resp.rstrip().endswith("세요.") or resp.rstrip().endswith("세요")
    scores["natural_ending"] = 1.0 if ends_natural else 0.5

    total = sum(scores.values()) / len(scores) if scores else 0
    return {"total": total, "details": scores}


def generate_response(model: str, query: str, max_tokens: int = 512) -> tuple:
    """Ollama에서 응답 생성, (response_text, latency_ms) 반환"""
    start = time.perf_counter()
    try:
        resp = httpx.post(OLLAMA_URL, json={
            "model": model,
            "prompt": query,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.4}
        }, timeout=120)
        text = resp.json().get("response", "").strip()
        latency = (time.perf_counter() - start) * 1000
        return text, latency
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return f"[ERROR: {e}]", latency


def main():
    print("=" * 70)
    print("  Base vs Fine-tuned 반복 평가 (10 prompts × 5 trials = n=50)")
    print(f"  실행 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Ollama 모델 확인
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        has_base = any(BASE_MODEL in m for m in models)
        has_ft = any(FINETUNED_MODEL in m for m in models)
        if not has_base or not has_ft:
            print(f"  [ERROR] 필요 모델 없음. 가용: {models}")
            return
    except Exception as e:
        print(f"  [ERROR] Ollama 연결 실패: {e}")
        return

    all_trials = []  # 모든 개별 결과

    total_runs = len(COMPARISON_PROMPTS) * NUM_TRIALS
    run_idx = 0

    for prompt_data in COMPARISON_PROMPTS:
        pid = prompt_data["id"]
        query = prompt_data["query"]
        intent = prompt_data["intent"]
        topic = prompt_data["topic"]
        print(f"\n{'─'*60}")
        print(f"  [{pid}] {topic}")
        print(f"  질문: {query}")

        for trial in range(1, NUM_TRIALS + 1):
            run_idx += 1
            print(f"    Trial {trial}/{NUM_TRIALS} ({run_idx}/{total_runs})...", end=" ", flush=True)

            base_resp, base_lat = generate_response(BASE_MODEL, query, max_tokens=200)
            ft_resp, ft_lat = generate_response(FINETUNED_MODEL, query, max_tokens=512)

            base_eval = auto_evaluate_response(query, base_resp, intent)
            ft_eval = auto_evaluate_response(query, ft_resp, intent)

            winner = "ft" if ft_eval["total"] > base_eval["total"] + 0.05 else (
                "base" if base_eval["total"] > ft_eval["total"] + 0.05 else "tie")

            print(f"Base={base_eval['total']:.3f} FT={ft_eval['total']:.3f} → {winner} "
                  f"(lat: {base_lat:.0f}/{ft_lat:.0f}ms)")

            all_trials.append({
                "prompt_id": pid,
                "topic": topic,
                "intent": intent,
                "trial": trial,
                "base_score": base_eval["total"],
                "ft_score": ft_eval["total"],
                "base_latency_ms": base_lat,
                "ft_latency_ms": ft_lat,
                "winner": winner,
                "base_response": base_resp,
                "ft_response": ft_resp,
                "base_details": base_eval["details"],
                "ft_details": ft_eval["details"],
            })

    # ── 집계 통계 ──
    print("\n" + "=" * 70)
    print("  집계 결과 (n={})".format(len(all_trials)))
    print("=" * 70)

    base_scores = [t["base_score"] for t in all_trials]
    ft_scores = [t["ft_score"] for t in all_trials]
    base_lats = [t["base_latency_ms"] for t in all_trials]
    ft_lats = [t["ft_latency_ms"] for t in all_trials]

    base_wins = sum(1 for t in all_trials if t["winner"] == "base")
    ft_wins = sum(1 for t in all_trials if t["winner"] == "ft")
    ties = sum(1 for t in all_trials if t["winner"] == "tie")

    print(f"\n  TABLE IV (확장): Base vs Fine-tuned (n={len(all_trials)})")
    print(f"  {'모델':<30} {'평균 품질':>10} {'표준편차':>10} {'평균 지연(ms)':>14}")
    print(f"  {'─'*64}")
    print(f"  {'Base SLM (Kanana 2.1B)':<30} {statistics.mean(base_scores):>10.3f} "
          f"{statistics.stdev(base_scores):>10.3f} {statistics.mean(base_lats):>14.0f}")
    print(f"  {'Fine-tuned SLM (LoRA)':<30} {statistics.mean(ft_scores):>10.3f} "
          f"{statistics.stdev(ft_scores):>10.3f} {statistics.mean(ft_lats):>14.0f}")
    print(f"  {'─'*64}")
    print(f"  승/패/무: Base {base_wins} / Fine-tuned {ft_wins} / Tie {ties}")

    # 프롬프트별 평균
    print(f"\n  프롬프트별 평균 점수 (각 {NUM_TRIALS}회):")
    print(f"  {'Topic':<30} {'Base (mean±std)':>18} {'FT (mean±std)':>18} {'Winner':>10}")
    print(f"  {'─'*76}")

    for prompt_data in COMPARISON_PROMPTS:
        pid = prompt_data["id"]
        topic = prompt_data["topic"]
        pt = [t for t in all_trials if t["prompt_id"] == pid]
        b = [t["base_score"] for t in pt]
        f_ = [t["ft_score"] for t in pt]
        b_mean, b_std = statistics.mean(b), statistics.stdev(b) if len(b) > 1 else 0
        f_mean, f_std = statistics.mean(f_), statistics.stdev(f_) if len(f_) > 1 else 0
        w = "FT" if f_mean > b_mean + 0.05 else ("Base" if b_mean > f_mean + 0.05 else "Tie")
        topic_short = topic[:28]
        print(f"  {topic_short:<30} {b_mean:>7.3f}±{b_std:.3f}  {f_mean:>7.3f}±{f_std:.3f}  {w:>10}")

    # 세부 지표별 평균
    print(f"\n  세부 지표별 평균 비교:")
    metric_names = set()
    for t in all_trials:
        metric_names.update(t["base_details"].keys())
    metric_names = sorted(metric_names)

    print(f"  {'Metric':<20} {'Base':>8} {'FT':>8} {'Diff':>8}")
    print(f"  {'─'*44}")
    for m in metric_names:
        b_vals = [t["base_details"].get(m, 0) for t in all_trials if m in t["base_details"]]
        f_vals = [t["ft_details"].get(m, 0) for t in all_trials if m in t["ft_details"]]
        if b_vals and f_vals:
            b_avg = statistics.mean(b_vals)
            f_avg = statistics.mean(f_vals)
            print(f"  {m:<20} {b_avg:>8.3f} {f_avg:>8.3f} {f_avg - b_avg:>+8.3f}")

    # ── JSON 저장 ──
    output_dir = Path(__file__).resolve().parents[2] / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"roman_model_comparison_n50_{timestamp}.json"

    summary = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_prompts": len(COMPARISON_PROMPTS),
            "num_trials": NUM_TRIALS,
            "total_n": len(all_trials),
            "base_model": BASE_MODEL,
            "finetuned_model": FINETUNED_MODEL,
        },
        "overall": {
            "base_mean": statistics.mean(base_scores),
            "base_std": statistics.stdev(base_scores),
            "ft_mean": statistics.mean(ft_scores),
            "ft_std": statistics.stdev(ft_scores),
            "base_latency_mean": statistics.mean(base_lats),
            "ft_latency_mean": statistics.mean(ft_lats),
            "base_wins": base_wins,
            "ft_wins": ft_wins,
            "ties": ties,
        },
        "trials": all_trials,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  결과 저장: {output_file}")


if __name__ == "__main__":
    main()
