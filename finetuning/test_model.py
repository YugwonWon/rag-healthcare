"""Fine-tuned EXAONE 모델 답변 품질 테스트 (결과를 파일로 저장)"""
import requests
import json
import os
import time

BASE = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(BASE, "output", "test_results_clean.txt")
OLLAMA_URL = "http://localhost:11434"
MODEL = "exaone-counseling"

TEST_CASES = [
    ("무릎 통증", "안녕하세요, 요즘 무릎이 시리고 아파요. 어떻게 하면 좋을까요?"),
    ("불면증", "밤에 잠이 잘 안 와서 걱정이에요. 어떻게 해야 할까요?"),
    ("치매 예방", "치매 예방하려면 어떻게 해야 하나요?"),
    ("혈압약 복용", "혈압약을 매일 먹는데 가끔 깜빡해요. 어떡하죠?"),
]


def save(lines):
    """중간 저장 (중단되어도 일부 결과 보존)"""
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        f.write("\n".join(lines))


def check_ollama():
    """Ollama 서버 상태 확인"""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        data = r.json()
        models = [m["name"] for m in data.get("models", [])]
        return True, models
    except Exception as e:
        return False, str(e)


def run_test(prompt, timeout=600):
    """단일 프롬프트 테스트"""
    start = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        elapsed = time.time() - start
        data = r.json()
        return data.get("response", "ERROR: no response"), elapsed
    except Exception as e:
        return f"ERROR: {e}", time.time() - start


def main():
    lines = []

    # 1. Ollama 상태 확인
    ok, info = check_ollama()
    lines.append("=== Ollama 상태 ===")
    if ok:
        lines.append(f"서버: 정상")
        lines.append(f"등록된 모델: {', '.join(info)}")
    else:
        lines.append(f"서버 오류: {info}")
        save(lines)
        return

    if not any(MODEL in m for m in info):
        lines.append(f"\n❌ '{MODEL}' 모델이 등록되어 있지 않습니다!")
        save(lines)
        return

    # 2. 모델 테스트
    lines.append(f"\n=== 모델 테스트: {MODEL} ===\n")

    for i, (title, prompt) in enumerate(TEST_CASES, 1):
        answer, elapsed = run_test(prompt)
        lines.append(f"--- 테스트 {i}: {title} ({elapsed:.1f}초) ---")
        lines.append(f"질문: {prompt}")
        lines.append(f"답변: {answer}")
        lines.append("")
        save(lines)  # 각 테스트마다 중간 저장

    lines.append("=== 테스트 완료 ===")
    save(lines)


if __name__ == "__main__":
    main()
