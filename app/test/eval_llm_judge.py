#!/usr/bin/env python3
"""
LLM-as-Judge 모델 비교 평가 (Gemini 2.5 Pro)
================================================
- 25개 프롬프트 (10 토픽 × 2~3 변형) × 3회 반복 = n=75
- Gemini 2.5 Pro를 judge로 활용한 5점 리커트 척도 평가
- 자동 heuristic 평가 병행
- 프롬프트별/토픽별 평균±표준편차 산출

사용법:
    cd /Users/yugwon/Projects/rag-healthcare
    .venv/bin/python3 -m app.test.eval_llm_judge
"""

import json
import re
import time
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import os

import httpx
from google import genai
from google.genai import types

# ── 설정 ──
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = "gemini-2.5-flash"
OLLAMA_URL = "http://localhost:11434/api/generate"
BASE_MODEL = "kanana-base-raw"
FINETUNED_MODEL = "kanana-counseling"
NUM_TRIALS = 3

# ── 확장된 프롬프트 (10 토픽 × 2~3 변형 = 25개) ──
PROMPTS = [
    # 수면장애 (3)
    {"id": "sleep_001", "intent": "health_consult", "topic": "수면장애",
     "query": "요즘 잠을 통 못 자겠어. 밤에 자다 깨고 또 깨고… 피곤해 죽겠어."},
    {"id": "sleep_002", "intent": "health_consult", "topic": "수면장애",
     "query": "새벽 3시만 되면 눈이 떠져. 다시 잠들기가 너무 힘들어."},
    {"id": "sleep_003", "intent": "health_consult", "topic": "수면장애",
     "query": "수면제 없이는 잠을 못 자는데, 계속 먹어도 괜찮을까?"},

    # 요실금/프라이버시 민감 (3)
    {"id": "incontinence_001", "intent": "health_consult", "topic": "요실금 (프라이버시 민감)",
     "query": "재채기만 하면 소변이 새는데, 너무 부끄러워서 아무한테도 말 못 하겠어."},
    {"id": "incontinence_002", "intent": "health_consult", "topic": "요실금 (프라이버시 민감)",
     "query": "밖에 나가면 화장실 걱정부터 해. 패드를 쓰는데도 불안해서…"},
    {"id": "incontinence_003", "intent": "health_consult", "topic": "요실금 (프라이버시 민감)",
     "query": "기침할 때마다 소변이 조금씩 나와서 창피해. 이게 나이 들면 다 그런 거야?"},

    # 인지기능 (3)
    {"id": "cognitive_001", "intent": "health_consult", "topic": "인지기능 (프라이버시 민감)",
     "query": "딸이 자꾸 기억력이 떨어진다고 걱정하더라고."},
    {"id": "cognitive_002", "intent": "health_consult", "topic": "인지기능 (프라이버시 민감)",
     "query": "방금 뭘 하려고 했는지 자꾸 까먹어. 물건도 어디 뒀는지 모르겠고."},
    {"id": "cognitive_003", "intent": "health_consult", "topic": "인지기능 (프라이버시 민감)",
     "query": "아들이 나보고 치매 검사 받으라는데... 나는 괜찮은 것 같은데."},

    # 복약 관리 (2)
    {"id": "medication_001", "intent": "medication", "topic": "복약 관리",
     "query": "혈압약을 며칠 안 먹었는데 괜찮을까?"},
    {"id": "medication_002", "intent": "medication", "topic": "복약 관리",
     "query": "약이 너무 많아서 뭐가 뭔지 모르겠어. 아침에만 다섯 알을 먹어야 해."},

    # 낙상 (2)
    {"id": "fall_001", "intent": "health_consult", "topic": "낙상",
     "query": "어제 화장실 가다가 미끄러져서 넘어졌어. 무릎이 아파."},
    {"id": "fall_002", "intent": "health_consult", "topic": "낙상",
     "query": "요즘 다리에 힘이 없어서 자꾸 비틀거려. 또 넘어질까 봐 무서워."},

    # 정서 건강 (3)
    {"id": "emotion_001", "intent": "health_consult", "topic": "정서 건강",
     "query": "요즘 뭘해도 재미가 없고 그냥 우울해."},
    {"id": "emotion_002", "intent": "health_consult", "topic": "정서 건강",
     "query": "혼자 있으면 외롭고, 눈물이 나올 때가 많아."},
    {"id": "emotion_003", "intent": "health_consult", "topic": "정서 건강",
     "query": "밤에 혼자 있으면 불안하고 무서워. 자식들한테 말하기도 미안하고."},

    # 일상 대화 (2)
    {"id": "general_001", "intent": "general_chat", "topic": "일상 대화",
     "query": "오늘 날씨가 좋으니까 기분이 좋아."},
    {"id": "general_002", "intent": "general_chat", "topic": "일상 대화",
     "query": "어제 손주가 와서 같이 밥 먹었어. 참 좋더라."},

    # 응급 (2)
    {"id": "emergency_001", "intent": "emergency", "topic": "응급",
     "query": "갑자기 가슴이 너무 아프고 숨이 안 쉬어져요."},
    {"id": "emergency_002", "intent": "emergency", "topic": "응급",
     "query": "머리가 갑자기 엄청 아프고 한쪽 팔이 안 움직여져요."},

    # 생활습관 (2)
    {"id": "lifestyle_001", "intent": "lifestyle", "topic": "생활습관",
     "query": "운동을 하고 싶은데 무릎이 안 좋아서 뭘 해야 할지 모르겠어."},
    {"id": "lifestyle_002", "intent": "lifestyle", "topic": "생활습관",
     "query": "혼자 살다 보니 밥을 대충 먹게 돼. 뭘 먹어야 좋을까?"},

    # 위생 (2 — 논문에서 위생 카테고리도 있으므로 추가)
    {"id": "hygiene_001", "intent": "health_consult", "topic": "위생",
     "query": "혼자 사니까 씻는 것도 귀찮아져. 목욕하기가 무서워 미끄러울까 봐."},
    {"id": "hygiene_002", "intent": "health_consult", "topic": "위생",
     "query": "이 닦는 걸 자꾸 깜빡해. 틀니 관리도 잘 안 되고."},
]


# ── Gemini LLM-as-Judge 프롬프트 ──
JUDGE_SYSTEM_PROMPT = """당신은 고령자 헬스케어 대화 품질을 평가하는 전문 평가자입니다.
고령자(65세 이상, 건강 문제가 있는)가 헬스케어 챗봇과 대화하는 상황입니다.

아래 5개 기준으로 각 응답을 1~5점(정수)으로 평가하고, 반드시 지정된 JSON 형식으로만 응답하세요.

평가 기준:
1. empathy (공감성): 사용자의 감정을 인정하고 공감하는가? (1=무관심/기계적, 5=깊은 공감)
2. appropriateness (적절성): 존댓말을 사용하고, 침습적 질문을 피하며, 판단하지 않는 어조인가? (1=무례/침습적, 5=매우 존중)
3. domain_knowledge (도메인 지식): 건강 관련 질문에 대해 정확하고 구체적인 조언을 제공하는가? (1=부정확/무관, 5=정확하고 구체적)
4. naturalness (자연스러움): 자연스러운 대화체인가? 목록/번호/마크다운 없이 할머니/할아버지와 대화하듯 자연스러운가? (1=기계적/형식적, 5=매우 자연스러움)
5. privacy_awareness (프라이버시 인식): 민감한 건강 주제에 대해 직접적/침습적 질문을 피하고, 사용자의 자발적 공개를 존중하는가? (1=침습적, 5=매우 배려)

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요:
{"empathy": N, "appropriateness": N, "domain_knowledge": N, "naturalness": N, "privacy_awareness": N, "reasoning": "한 줄 근거"}"""


JUDGE_USER_TEMPLATE = """[사용자 질문]
{query}

[응답 A (Base 모델)]
{response_a}

[응답 B (Fine-tuned 모델)]
{response_b}

위 두 응답을 각각 평가하세요. 반드시 아래 JSON 형식으로만 응답하세요:
{{"response_a": {{"empathy": N, "appropriateness": N, "domain_knowledge": N, "naturalness": N, "privacy_awareness": N}}, "response_b": {{"empathy": N, "appropriateness": N, "domain_knowledge": N, "naturalness": N, "privacy_awareness": N}}, "reasoning": "한 줄 비교 근거"}}"""


# Gemini client (초기화는 main에서)
client: genai.Client = None


def gemini_judge(query: str, resp_a: str, resp_b: str) -> Optional[Dict]:
    """Gemini로 두 응답을 비교 평가 (JSON 모드 강제)"""
    prompt = JUDGE_USER_TEMPLATE.format(query=query, response_a=resp_a, response_b=resp_b)
    full_prompt = JUDGE_SYSTEM_PROMPT + "\n\n" + prompt

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,
                response_mime_type="application/json",
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ],
            ),
        )
        text = response.text
        if text is None:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    text = part.text
                    break
        if not text:
            print(f"    [JUDGE ERROR] Empty response")
            return None
        text = text.strip()
        return json.loads(text)
    except Exception as e:
        print(f"    [JUDGE ERROR] {e}")
        return None


def generate_response(model: str, query: str, max_tokens: int = 512) -> tuple:
    """Ollama 응답 생성"""
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


def heuristic_evaluate(resp: str, intent: str) -> Dict:
    """기존 heuristic 자동 평가 (보조 지표)"""
    scores = {}
    length = len(resp)
    scores["length"] = 1.0 if 20 <= length <= 300 else (0.3 if length < 20 else 0.7)

    honorific_endings = ["요", "다", "세요", "니다", "습니다", "어요", "아요"]
    scores["honorific"] = 1.0 if any(resp.rstrip().endswith(e) for e in honorific_endings) else 0.5

    empathy_words = ["이해", "걱정", "힘드", "괴로", "불편", "안타깝", "많이", "그러시",
                     "아이고", "속상", "고생", "마음"]
    scores["empathy"] = 1.0 if any(w in resp for w in empathy_words) else 0.0

    has_list = bool(re.search(r'^\s*[\d\-\*·•][\.\)]\s', resp, re.MULTILINE))
    has_markdown = bool(re.search(r'\*\*[^*]+\*\*', resp))
    has_numbered = bool(re.search(r'^\s*\d+[\.\)]\s', resp, re.MULTILINE))
    scores["no_list"] = 0.0 if (has_list or has_markdown or has_numbered) else 1.0

    forbidden = ["도움이 되셨", "도움이 되었", "궁금한 점이 있으시면", "추가 질문"]
    scores["no_forbidden"] = 0.0 if any(f in resp for f in forbidden) else 1.0

    if intent == "emergency":
        scores["emergency_119"] = 1.0 if "119" in resp else 0.0

    invasive = ["치매입니까", "치매 증상", "요실금이 있으신가요", "우울증이 있으신가요"]
    scores["privacy"] = 0.0 if any(inv in resp for inv in invasive) else 1.0

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

    ends_natural = resp.rstrip().endswith("?") or resp.rstrip().endswith("요?") or \
                   resp.rstrip().endswith("세요.") or resp.rstrip().endswith("세요")
    scores["natural_ending"] = 1.0 if ends_natural else 0.5

    total = sum(scores.values()) / len(scores) if scores else 0
    return {"total": total, "details": scores}


def main():
    print("=" * 70)
    print("  LLM-as-Judge 모델 비교 평가")
    print(f"  Judge: {GEMINI_MODEL}")
    print(f"  프롬프트: {len(PROMPTS)}개 × {NUM_TRIALS}회 = n={len(PROMPTS)*NUM_TRIALS}")
    print(f"  실행 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Gemini 초기화
    global client
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Ollama 모델 확인
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        if not any(BASE_MODEL in m for m in models) or not any(FINETUNED_MODEL in m for m in models):
            print("[ERROR] Ollama 모델 없음")
            return
    except Exception as e:
        print(f"[ERROR] Ollama 연결 실패: {e}")
        return

    all_trials = []
    total_runs = len(PROMPTS) * NUM_TRIALS
    run_idx = 0
    judge_failures = 0

    for prompt_data in PROMPTS:
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

            # 모델 응답 생성
            base_resp, base_lat = generate_response(BASE_MODEL, query, 200)
            ft_resp, ft_lat = generate_response(FINETUNED_MODEL, query, 512)

            # Heuristic 평가
            base_heur = heuristic_evaluate(base_resp, intent)
            ft_heur = heuristic_evaluate(ft_resp, intent)

            # LLM Judge 평가
            judge_result = gemini_judge(query, base_resp, ft_resp)

            if judge_result:
                ja = judge_result.get("response_a", {})
                jb = judge_result.get("response_b", {})
                base_judge_avg = statistics.mean(ja.values()) if ja and all(isinstance(v, (int, float)) for v in ja.values()) else None
                ft_judge_avg = statistics.mean(jb.values()) if jb and all(isinstance(v, (int, float)) for v in jb.values()) else None
                reasoning = judge_result.get("reasoning", "")
            else:
                ja, jb = {}, {}
                base_judge_avg = ft_judge_avg = None
                reasoning = "judge_error"
                judge_failures += 1

            # 로그 출력
            bj = f"{base_judge_avg:.1f}" if base_judge_avg else "ERR"
            fj = f"{ft_judge_avg:.1f}" if ft_judge_avg else "ERR"
            print(f"Heur: {base_heur['total']:.3f}/{ft_heur['total']:.3f} | "
                  f"Judge: {bj}/{fj} | lat: {base_lat:.0f}/{ft_lat:.0f}ms")

            trial_data = {
                "prompt_id": pid, "topic": topic, "intent": intent, "trial": trial,
                "query": query,
                "base_response": base_resp, "ft_response": ft_resp,
                "base_latency_ms": base_lat, "ft_latency_ms": ft_lat,
                "heuristic": {"base": base_heur, "ft": ft_heur},
                "judge": {
                    "base_scores": ja, "ft_scores": jb,
                    "base_avg": base_judge_avg, "ft_avg": ft_judge_avg,
                    "reasoning": reasoning,
                },
            }
            all_trials.append(trial_data)

            # Gemini rate limit 대응 (짧은 대기)
            time.sleep(2)

    # ═══ 집계 ═══
    print("\n" + "=" * 70)
    print(f"  집계 결과 (n={len(all_trials)}, judge 실패: {judge_failures})")
    print("=" * 70)

    # Judge 유효 데이터만
    valid = [t for t in all_trials if t["judge"]["base_avg"] is not None]
    n_valid = len(valid)

    if n_valid == 0:
        print("  [ERROR] Judge 평가 데이터 없음")
        save_results(all_trials, judge_failures, {}, {})
        return

    # 전체 Judge 점수
    base_j = [t["judge"]["base_avg"] for t in valid]
    ft_j = [t["judge"]["ft_avg"] for t in valid]
    base_h = [t["heuristic"]["base"]["total"] for t in all_trials]
    ft_h = [t["heuristic"]["ft"]["total"] for t in all_trials]
    base_lats = [t["base_latency_ms"] for t in all_trials]
    ft_lats = [t["ft_latency_ms"] for t in all_trials]

    # Judge 승/패/무 (0.3점 이상 차이)
    j_base_wins = sum(1 for t in valid if t["judge"]["base_avg"] > t["judge"]["ft_avg"] + 0.3)
    j_ft_wins = sum(1 for t in valid if t["judge"]["ft_avg"] > t["judge"]["base_avg"] + 0.3)
    j_ties = n_valid - j_base_wins - j_ft_wins

    print(f"\n  ┌─ TABLE IV-A: LLM-as-Judge 평가 (Gemini 2.5 Pro, n={n_valid})")
    print(f"  │ {'모델':<28} {'평균':>6} {'SD':>6} {'평균 지연':>10}")
    print(f"  │ {'─'*52}")
    print(f"  │ {'Base SLM (Kanana 2.1B)':<28} {statistics.mean(base_j):>6.2f} "
          f"{statistics.stdev(base_j):>6.2f} {statistics.mean(base_lats):>10.0f}ms")
    print(f"  │ {'Fine-tuned SLM (LoRA)':<28} {statistics.mean(ft_j):>6.2f} "
          f"{statistics.stdev(ft_j):>6.2f} {statistics.mean(ft_lats):>10.0f}ms")
    print(f"  │ 승/패/무: Base {j_base_wins} / FT {j_ft_wins} / Tie {j_ties}")
    print(f"  └{'─'*55}")

    print(f"\n  ┌─ TABLE IV-B: Heuristic 자동 평가 (n={len(all_trials)})")
    print(f"  │ {'모델':<28} {'평균':>6} {'SD':>6}")
    print(f"  │ {'─'*42}")
    print(f"  │ {'Base SLM (Kanana 2.1B)':<28} {statistics.mean(base_h):>6.3f} "
          f"{statistics.stdev(base_h):>6.3f}")
    print(f"  │ {'Fine-tuned SLM (LoRA)':<28} {statistics.mean(ft_h):>6.3f} "
          f"{statistics.stdev(ft_h):>6.3f}")
    print(f"  └{'─'*45}")

    # Judge 기준별 점수
    print(f"\n  세부 Judge 기준별 평균 (5점 척도):")
    criteria = ["empathy", "appropriateness", "domain_knowledge", "naturalness", "privacy_awareness"]
    print(f"  {'Criterion':<22} {'Base':>6} {'FT':>6} {'Diff':>7}")
    print(f"  {'─'*42}")
    criteria_stats = {}
    for c in criteria:
        b_vals = [t["judge"]["base_scores"].get(c, 0) for t in valid
                  if isinstance(t["judge"]["base_scores"].get(c), (int, float))]
        f_vals = [t["judge"]["ft_scores"].get(c, 0) for t in valid
                  if isinstance(t["judge"]["ft_scores"].get(c), (int, float))]
        if b_vals and f_vals:
            b_avg = statistics.mean(b_vals)
            f_avg = statistics.mean(f_vals)
            print(f"  {c:<22} {b_avg:>6.2f} {f_avg:>6.2f} {f_avg - b_avg:>+7.2f}")
            criteria_stats[c] = {"base": b_avg, "ft": f_avg, "diff": f_avg - b_avg}

    # 토픽별 Judge 평균
    topics = sorted(set(p["topic"] for p in PROMPTS))
    print(f"\n  토픽별 Judge 평균:")
    print(f"  {'Topic':<30} {'Base':>6} {'FT':>6} {'Winner':>10}")
    print(f"  {'─'*52}")
    topic_stats = {}
    for topic in topics:
        bt = [t["judge"]["base_avg"] for t in valid if t["topic"] == topic]
        ft = [t["judge"]["ft_avg"] for t in valid if t["topic"] == topic]
        if bt and ft:
            bm, fm = statistics.mean(bt), statistics.mean(ft)
            w = "FT" if fm > bm + 0.3 else ("Base" if bm > fm + 0.3 else "Tie")
            t_short = topic[:28]
            print(f"  {t_short:<30} {bm:>6.2f} {fm:>6.2f} {w:>10}")
            topic_stats[topic] = {"base": bm, "ft": fm, "winner": w}

    save_results(all_trials, judge_failures, criteria_stats, topic_stats)


def save_results(all_trials, judge_failures, criteria_stats, topic_stats):
    """JSON 저장"""
    valid = [t for t in all_trials if t["judge"]["base_avg"] is not None]
    base_j = [t["judge"]["base_avg"] for t in valid] if valid else [0]
    ft_j = [t["judge"]["ft_avg"] for t in valid] if valid else [0]
    base_h = [t["heuristic"]["base"]["total"] for t in all_trials]
    ft_h = [t["heuristic"]["ft"]["total"] for t in all_trials]

    output_dir = Path(__file__).resolve().parents[2] / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"roman_llm_judge_{timestamp}.json"

    summary = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "judge_model": GEMINI_MODEL,
            "num_prompts": len(PROMPTS),
            "num_trials": NUM_TRIALS,
            "total_n": len(all_trials),
            "valid_judge_n": len(valid),
            "judge_failures": judge_failures,
        },
        "judge_overall": {
            "base_mean": statistics.mean(base_j),
            "base_std": statistics.stdev(base_j) if len(base_j) > 1 else 0,
            "ft_mean": statistics.mean(ft_j),
            "ft_std": statistics.stdev(ft_j) if len(ft_j) > 1 else 0,
        },
        "heuristic_overall": {
            "base_mean": statistics.mean(base_h),
            "base_std": statistics.stdev(base_h) if len(base_h) > 1 else 0,
            "ft_mean": statistics.mean(ft_h),
            "ft_std": statistics.stdev(ft_h) if len(ft_h) > 1 else 0,
        },
        "criteria_stats": criteria_stats,
        "topic_stats": topic_stats,
        "trials": all_trials,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {output_file}")


if __name__ == "__main__":
    main()
