#!/usr/bin/env python3
"""음성 채팅 WebSocket(/ws/voice-chat) 동시 부하 테스트.

수업 상황(20~30명 동시 사용)에서 어디가 먼저 막히는지 측정한다. 텍스트 턴
({"type":"text",...})만 보내도 서버에서는 LLM 생성 + 문장별 TTS 합성이 그대로
돌기 때문에, GPU를 공유하는 주 병목(Ollama·MeloTTS)을 STT 녹음 없이 재현할 수 있다.

각 가상 사용자는 WS로 접속해 N개의 턴을 순차적으로 주고받으며 다음을 측정한다:
  - connect: 접속 성립까지(ms)
  - ttfb   : 메시지 전송 → 첫 text_chunk 도착까지(ms, 체감 응답성)
  - turn   : 메시지 전송 → done 도착까지(ms)
  - audio  : 받은 오디오(WAV) 청크 수/총 바이트
  - error/timeout 카운트

사용 예:
  .venv/bin/python scripts/loadtest_voice_ws.py --concurrent 25 --turns 3
  .venv/bin/python scripts/loadtest_voice_ws.py -c 30 --url ws://localhost:8000 --ramp 5
  .venv/bin/python scripts/loadtest_voice_ws.py -c 25 --url wss://yugwon-macmini.tail7f37ba.ts.net

종료 코드: 에러/타임아웃이 하나라도 있으면 1, 전부 성공이면 0.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import websockets
except ImportError:  # pragma: no cover
    raise SystemExit(
        "websockets 모듈이 필요합니다. 프로젝트 venv로 실행하세요:\n"
        "  .venv/bin/python scripts/loadtest_voice_ws.py ..."
    )

# 어르신이 던질 법한 다양한 질문(인텐트가 골고루 섞이도록 — 일반/건강/약물/생활).
DEFAULT_PROMPTS = [
    "안녕하세요, 오늘 날씨가 참 좋네요.",
    "요즘 무릎이 자주 아픈데 어떻게 하면 좋을까요?",
    "혈압약을 깜빡하고 안 먹었는데 지금 먹어도 되나요?",
    "밤에 잠이 잘 안 와요. 어떻게 해야 할까요?",
    "운동을 시작하고 싶은데 뭐부터 하면 좋을까요?",
    "기억력이 자꾸 떨어지는 것 같아 걱정이에요.",
]


@dataclass
class TurnResult:
    ttfb_ms: Optional[float] = None
    turn_ms: Optional[float] = None
    audio_chunks: int = 0
    audio_bytes: int = 0
    error: Optional[str] = None


@dataclass
class SessionResult:
    user_id: int
    connect_ms: Optional[float] = None
    connect_error: Optional[str] = None
    turns: List[TurnResult] = field(default_factory=list)


async def run_session(
    user_id: int,
    base_url: str,
    n_turns: int,
    prompts: List[str],
    open_timeout: float,
    turn_timeout: float,
    start_delay: float,
) -> SessionResult:
    res = SessionResult(user_id=user_id)
    if start_delay:
        await asyncio.sleep(start_delay)

    nickname = f"load{user_id:03d}"
    uri = f"{base_url}/ws/voice-chat?nickname={nickname}"
    t0 = time.perf_counter()
    try:
        ws = await asyncio.wait_for(
            websockets.connect(uri, max_size=None, ping_interval=20),
            timeout=open_timeout,
        )
    except Exception as e:  # noqa: BLE001
        res.connect_error = f"{type(e).__name__}: {e}"
        return res
    res.connect_ms = (time.perf_counter() - t0) * 1000

    try:
        for i in range(n_turns):
            prompt = prompts[(user_id + i) % len(prompts)]
            tr = TurnResult()
            sent = time.perf_counter()
            try:
                await ws.send(json.dumps({"type": "text", "text": prompt}))
                tr = await asyncio.wait_for(
                    _collect_turn(ws, sent), timeout=turn_timeout
                )
            except asyncio.TimeoutError:
                tr.error = f"timeout>{turn_timeout:.0f}s"
            except Exception as e:  # noqa: BLE001
                tr.error = f"{type(e).__name__}: {e}"
            res.turns.append(tr)
            if tr.error:
                break  # 한 번 막히면 그 세션은 중단(현실에서도 사용자가 이탈)
    finally:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
    return res


async def _collect_turn(ws, sent: float) -> TurnResult:
    """한 턴 동안 done 이 올 때까지 메시지를 모은다."""
    tr = TurnResult()
    while True:
        raw = await ws.recv()
        if isinstance(raw, (bytes, bytearray)):
            tr.audio_chunks += 1
            tr.audio_bytes += len(raw)
            continue
        msg = json.loads(raw)
        mtype = msg.get("type")
        if mtype == "text_chunk":
            if tr.ttfb_ms is None:
                tr.ttfb_ms = (time.perf_counter() - sent) * 1000
        elif mtype == "transcript":
            continue
        elif mtype == "error":
            tr.error = "server:" + str(msg.get("detail", ""))[:80]
            tr.turn_ms = (time.perf_counter() - sent) * 1000
            return tr
        elif mtype == "done":
            tr.turn_ms = (time.perf_counter() - sent) * 1000
            return tr


def _pct(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _fmt(values: List[float], unit: str = "ms") -> str:
    if not values:
        return "  (없음)"
    return (
        f"n={len(values)}  p50={_pct(values,50):.0f}  p95={_pct(values,95):.0f}  "
        f"max={max(values):.0f}  avg={statistics.mean(values):.0f} {unit}"
    )


def summarize(results: List[SessionResult], wall_s: float, args) -> int:
    connect_ok = [r.connect_ms for r in results if r.connect_ms is not None]
    connect_err = [r for r in results if r.connect_error]

    all_turns: List[TurnResult] = [t for r in results for t in r.turns]
    ok_turns = [t for t in all_turns if not t.error]
    err_turns = [t for t in all_turns if t.error]

    ttfb = [t.ttfb_ms for t in ok_turns if t.ttfb_ms is not None]
    turn = [t.turn_ms for t in ok_turns if t.turn_ms is not None]
    audio_chunks = [t.audio_chunks for t in ok_turns]
    audio_ok = sum(1 for t in ok_turns if t.audio_chunks > 0)

    print("\n" + "=" * 64)
    print(f"  부하 테스트 결과 — 동시 {args.concurrent}명 × {args.turns}턴")
    print(f"  대상: {args.url}   총 소요: {wall_s:.1f}s")
    print("=" * 64)
    print(f"\n■ 접속(connect)")
    print(f"  성공 {len(connect_ok)}/{len(results)}   " + _fmt(connect_ok))
    if connect_err:
        print(f"  ❌ 접속 실패 {len(connect_err)}건:")
        for r in connect_err[:5]:
            print(f"     user{r.user_id:03d}: {r.connect_error}")

    print(f"\n■ 첫 응답까지(ttfb, 체감 응답성)")
    print("  " + _fmt(ttfb))
    print(f"\n■ 턴 완료까지(turn, 전체 답변)")
    print("  " + _fmt(turn))

    print(f"\n■ 오디오(TTS)")
    print(f"  오디오 받은 턴 {audio_ok}/{len(ok_turns)}   "
          f"평균 청크 {statistics.mean(audio_chunks):.1f}" if ok_turns else "  (성공 턴 없음)")

    print(f"\n■ 실패/지연")
    print(f"  성공 턴 {len(ok_turns)}   실패 턴 {len(err_turns)}")
    if err_turns:
        from collections import Counter
        kinds = Counter(t.error.split(":")[0] for t in err_turns)
        for k, v in kinds.most_common():
            print(f"     {k}: {v}건")
        for t in err_turns[:5]:
            print(f"     예: {t.error}")

    # 처리량
    if turn:
        thru = len(ok_turns) / wall_s
        print(f"\n■ 처리량  ≈ {thru:.2f} 턴/초  ({thru*60:.0f} 턴/분)")

    print("\n" + "=" * 64)
    failed = len(connect_err) + len(err_turns)
    if failed == 0:
        print("  ✅ 전부 성공 — 이 동시성에서는 여유가 있습니다.")
    else:
        print(f"  ⚠️  실패 {failed}건 — 동시성을 낮추거나 대비책 필요(아래 권장).")
        print("     · Ollama: OLLAMA_NUM_PARALLEL / KEEP_ALIVE 설정")
        print("     · 앱 동시성 캡 + 대기 안내, 사이드카 타임아웃 상향")
    print("=" * 64 + "\n")
    return 1 if failed else 0


async def main_async(args) -> int:
    base_url = args.url.rstrip("/")
    prompts = DEFAULT_PROMPTS
    print(f"▶ 시작: {args.concurrent}명 동시 접속, 각 {args.turns}턴 "
          f"(ramp {args.ramp}s, 턴 타임아웃 {args.turn_timeout}s)")
    print(f"  대상 {base_url}/ws/voice-chat\n")

    ramp_step = (args.ramp / max(1, args.concurrent - 1)) if args.concurrent > 1 else 0
    t0 = time.perf_counter()
    tasks = [
        asyncio.create_task(
            run_session(
                i, base_url, args.turns, prompts,
                args.open_timeout, args.turn_timeout, i * ramp_step,
            )
        )
        for i in range(args.concurrent)
    ]
    results = await asyncio.gather(*tasks)
    wall_s = time.perf_counter() - t0
    return summarize(list(results), wall_s, args)


def parse_args():
    p = argparse.ArgumentParser(description="음성 WS 동시 부하 테스트")
    p.add_argument("--url", default="ws://localhost:8000",
                   help="WS 베이스 URL (기본 ws://localhost:8000; 원격은 wss://...)")
    p.add_argument("-c", "--concurrent", type=int, default=25, help="동시 사용자 수")
    p.add_argument("-t", "--turns", type=int, default=3, help="사용자당 턴 수")
    p.add_argument("--ramp", type=float, default=3.0,
                   help="전 사용자 접속을 이 시간(초)에 걸쳐 분산(0=완전 동시)")
    p.add_argument("--open-timeout", type=float, default=15.0, help="접속 타임아웃(초)")
    p.add_argument("--turn-timeout", type=float, default=60.0,
                   help="한 턴 응답 타임아웃(초) — 앱 httpx 타임아웃(60s)과 맞춤")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        raise SystemExit(asyncio.run(main_async(args)))
    except KeyboardInterrupt:
        print("\n중단됨.")
        raise SystemExit(130)
