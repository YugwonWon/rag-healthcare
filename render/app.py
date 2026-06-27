"""
Gradio 프론트엔드 애플리케이션
Render 배포용
"""

import os
import re
import tempfile
import httpx
import pandas as pd
import gradio as gr
from datetime import datetime
from typing import Optional
from packaging import version
from deep_translator import GoogleTranslator


def translate_to_english(text: str) -> str:
    """한국어 텍스트를 영어로 번역"""
    if not text or not text.strip():
        return text
    try:
        translator = GoogleTranslator(source='ko', target='en')
        return translator.translate(text)
    except Exception as e:
        print(f"번역 오류: {e}")
        return ""


def translate_line_by_line(text: str) -> str:
    """각 줄마다 (영어 번역) 형식으로 변환"""
    if not text or not text.strip():
        return text
    
    lines = text.split('\n')
    result_lines = []
    translator = GoogleTranslator(source='ko', target='en')
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            result_lines.append(line)
            continue
        
        # 이모지나 특수문자만 있는 줄은 번역하지 않음
        # 한글이 포함된 경우에만 번역
        import re
        if not re.search(r'[가-힣]', stripped):
            result_lines.append(line)
            continue
        
        try:
            translated = translator.translate(stripped)
            result_lines.append(f"{line}\n({translated})")
        except Exception as e:
            print(f"줄 번역 오류: {e}")
            result_lines.append(line)
    
    return '\n'.join(result_lines)


def translate_to_korean(text: str) -> str:
    """영어 텍스트를 한국어로 번역"""
    if not text or not text.strip():
        return text
    try:
        translator = GoogleTranslator(source='en', target='ko')
        return translator.translate(text)
    except Exception as e:
        print(f"번역 오류: {e}")
        return text

# Gradio 버전 감지
GRADIO_VERSION = version.parse(gr.__version__)
IS_GRADIO_5 = GRADIO_VERSION < version.parse("6.0.0")
print(f"📦 Gradio 버전: {gr.__version__} (5.x: {IS_GRADIO_5})")

# 환경변수에서 백엔드 URL 가져오기
BACKEND_URL = os.getenv("BACKEND_URL", "https://yugwon-macmini.tail7f37ba.ts.net")

# 공통 요청 헤더 — CLIENT_API_KEY 설정 시 X-API-Key 포함 (백엔드 게이트 통과)
_client_api_key = os.getenv("CLIENT_API_KEY", "")
BASE_HEADERS: dict = {"ngrok-skip-browser-warning": "true"}
if _client_api_key:
    BASE_HEADERS["X-API-Key"] = _client_api_key

# 상태 저장용
user_sessions = {}

print(f"🔗 Backend URL: {BACKEND_URL}")
print(f"🔑 API Key: {'설정됨' if _client_api_key else '미설정'}")

# Cloud Run 콜드 스타트 + LLM 응답 시간 고려 (최대 180초)
API_TIMEOUT = 180.0


async def call_api(endpoint: str, method: str = "GET", data: Optional[dict] = None) -> dict:
    """백엔드 API 호출"""
    async with httpx.AsyncClient(timeout=API_TIMEOUT, headers=BASE_HEADERS) as client:
        url = f"{BACKEND_URL}{endpoint}"
        
        try:
            print(f"📡 API 호출: {method} {url}")
            if method == "GET":
                response = await client.get(url, params=data)
            else:
                response = await client.post(url, json=data)
            
            response.raise_for_status()
            return response.json()
        
        except httpx.TimeoutException:
            print(f"⏰ API 타임아웃: {url}")
            return {"error": "서버 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."}
        
        except httpx.HTTPError as e:
            print(f"❌ API 에러: {e}")
            return {"error": str(e)}


async def check_backend_status() -> str:
    """백엔드 및 모델 상태 확인"""
    try:
        result = await call_api("/health", "GET")
        if "error" in result:
            return "🔴 서버 연결 실패"
        
        llm_available = result.get("llm_available", False)
        stats = result.get("stats", {})
        doc_count = stats.get("documents", 0)
        
        if llm_available:
            return f"🟢 모델 준비됨 (문서: {doc_count}개)"
        else:
            return "🟡 모델 로딩 중..."
    except Exception as e:
        print(f"❌ 상태 확인 에러: {e}")
        return "🔴 서버 연결 실패"


async def delete_conversation_history(nickname: str) -> str:
    """대화 기록 삭제"""
    if not nickname.strip():
        return "❌ 닉네임을 먼저 입력해주세요."
    
    # DELETE 메서드 호출
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        try:
            url = f"{BACKEND_URL}/history/{nickname}"
            response = await client.delete(url)
            response.raise_for_status()
            result = response.json()
            
            deleted_count = result.get("deleted_count", 0)
            return f"✅ {nickname}님의 대화 기록 {deleted_count}개가 삭제되었습니다."
        except httpx.HTTPError as e:
            print(f"❌ 대화 기록 삭제 에러: {e}")
            return f"❌ 삭제 실패: {str(e)}"


# ==========================================
# 관리자 — 대화 기록 CSV 다운로드 (교수/연구자용)
# ==========================================

async def admin_login(password: str):
    """관리자 비밀번호 검증. 성공 시 다운로드 영역을 노출한다."""
    if not password or not password.strip():
        return "비밀번호를 입력해주세요.", gr.update(visible=False), ""
    headers = {**BASE_HEADERS, "X-Admin-Password": password}
    async with httpx.AsyncClient(timeout=API_TIMEOUT, headers=headers) as client:
        try:
            resp = await client.post(f"{BACKEND_URL}/admin/login")
            if resp.status_code == 200:
                return "✅ 로그인 성공. 아래에서 대화 기록을 다운로드하세요.", gr.update(visible=True), password
            if resp.status_code == 401:
                return "❌ 비밀번호가 올바르지 않습니다.", gr.update(visible=False), ""
            return f"❌ 로그인 실패 (status {resp.status_code})", gr.update(visible=False), ""
        except httpx.HTTPError as e:
            print(f"❌ 관리자 로그인 에러: {e}")
            return f"❌ 서버 연결 실패: {e}", gr.update(visible=False), ""


async def admin_download_csv(password: str, source_label: str):
    """대화 기록 CSV를 받아 임시파일로 저장하고 다운로드용 파일로 노출한다."""
    if not password:
        return None, "먼저 로그인해주세요."
    source = "logs" if "분석" in (source_label or "") else "history"
    headers = {**BASE_HEADERS, "X-Admin-Password": password}
    async with httpx.AsyncClient(timeout=API_TIMEOUT, headers=headers) as client:
        try:
            resp = await client.get(
                f"{BACKEND_URL}/admin/conversations.csv",
                params={"source": source},
            )
            if resp.status_code == 401:
                return None, "❌ 인증이 만료되었습니다. 다시 로그인해주세요."
            resp.raise_for_status()

            cd = resp.headers.get("content-disposition", "")
            m = re.search(r'filename="?([^"]+)"?', cd)
            fname = m.group(1) if m else f"conversations_{source}.csv"

            tmp_dir = tempfile.mkdtemp()
            path = os.path.join(tmp_dir, fname)
            with open(path, "wb") as f:
                f.write(resp.content)

            # CSV 행 수(헤더 제외) 대략 표기
            line_count = max(resp.content.count(b"\n") - 1, 0)
            return path, f"✅ {fname} 준비 완료 (약 {line_count}행)"
        except httpx.HTTPError as e:
            print(f"❌ 관리자 CSV 다운로드 에러: {e}")
            return None, f"❌ 다운로드 실패: {e}"


async def admin_view_table(password: str, source_label: str):
    """대화 기록을 웹에서 표(컬럼) 형태로 조회한다."""
    if not password:
        return pd.DataFrame(), "먼저 로그인해주세요."
    source = "logs" if "분석" in (source_label or "") else "history"
    headers = {**BASE_HEADERS, "X-Admin-Password": password}
    async with httpx.AsyncClient(timeout=API_TIMEOUT, headers=headers) as client:
        try:
            resp = await client.get(
                f"{BACKEND_URL}/admin/conversations.json",
                params={"source": source},
            )
            if resp.status_code == 401:
                return pd.DataFrame(), "❌ 인증이 만료되었습니다. 다시 로그인해주세요."
            resp.raise_for_status()
            data = resp.json()
            cols = data.get("columns", [])
            rows = data.get("rows", [])
            df = pd.DataFrame(rows, columns=cols) if cols else pd.DataFrame()
            total, shown = data.get("total", 0), data.get("shown", 0)
            if total == 0:
                msg = "표시할 대화 기록이 없습니다."
                if source == "logs":
                    msg += " (분석 로그는 백엔드 재시작 이후 쌓인 대화부터 기록됩니다.)"
                return df, msg
            note = f"총 {total}행 중 최근 {shown}행 표시" if shown < total else f"총 {total}행 표시"
            return df, f"✅ {note}"
        except httpx.HTTPError as e:
            print(f"❌ 관리자 표 조회 에러: {e}")
            return pd.DataFrame(), f"❌ 조회 실패: {e}"


async def get_greeting(nickname: str) -> str:
    """개인화된 인사말 가져오기"""
    result = await call_api("/greeting", "POST", {"nickname": nickname})
    
    if "error" in result:
        return f"안녕하세요, {nickname}님! 오늘도 좋은 하루 되세요. 😊"
    
    greeting = result.get("greeting", "")
    suggestions = result.get("suggestions", [])
    
    response = greeting
    if suggestions:
        response += "\n\n💡 오늘 이런 활동은 어떠세요?\n"
        response += "\n".join(f"  • {s}" for s in suggestions[:3])
    
    return response


async def get_profile(nickname: str) -> dict:
    """저장된 프로필 불러오기"""
    result = await call_api(f"/profile/{nickname}", "GET")
    
    if "error" in result:
        return {}
    
    return result.get("profile", {})


async def chat_with_bot(
    nickname: str,
    message: str,
    history: list,
    enable_translation: bool = False,
) -> tuple[list, str]:
    """챗봇과 대화"""
    if not nickname.strip():
        return history, "닉네임을 먼저 입력해주세요."
    
    if not message.strip():
        return history, ""
    
    # 첫 대화인 경우 프로필 저장
    if nickname not in user_sessions:
        await call_api("/profile", "POST", {"nickname": nickname})
        user_sessions[nickname] = {"started_at": datetime.now().isoformat()}
    
    # 채팅 API 호출
    result = await call_api("/chat", "POST", {
        "nickname": nickname,
        "message": message,
        "include_history": True
    })
    
    if "error" in result:
        bot_response = f"죄송합니다, 일시적인 오류가 발생했어요. 다시 말씀해 주세요. 🙏"
    else:
        bot_response = result.get("response", "응답을 받지 못했습니다.")
        
        # 증상 알림이 있으면 추가
        symptom_alert = result.get("symptom_alert")
        if symptom_alert and symptom_alert.get("needs_attention"):
            bot_response += "\n\n⚠️ **주의가 필요합니다**"
            for rec in symptom_alert.get("recommendations", []):
                bot_response += f"\n  • {rec}"
        
        # 복약 알림이 있으면 추가
        med_reminders = result.get("medication_reminders")
        if med_reminders:
            bot_response += "\n\n" + "\n".join(med_reminders)
    
    # 대화 기록 업데이트 (Gradio 6.x 형식)
    if enable_translation:
        # 번역 모드: 영어로만 표시
        user_display = translate_to_english(message)
        bot_display = translate_to_english(bot_response)
        history.append({"role": "user", "content": user_display})
        history.append({"role": "assistant", "content": bot_display})
    else:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_response})
    
    return history, ""


async def _synth_tts(response: str):
    """텍스트 → (tts_path, data_url). JS 가 #tts_player 의 data URL 로 직접 재생한다."""
    tts_path, tts_data_url = None, ""
    if not response:
        return tts_path, tts_data_url
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT, headers=BASE_HEADERS) as client:
            tr = await client.post(f"{BACKEND_URL}/tts", json={"text": response})
        if tr.status_code == 200:
            ctype = tr.headers.get("content-type", "audio/wav")
            is_mp3 = "mpeg" in ctype or "mp3" in ctype
            ext = ".mp3" if is_mp3 else ".wav"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                tf.write(tr.content)
                tts_path = tf.name
            import base64 as _b64
            mime = "audio/mpeg" if is_mp3 else "audio/wav"
            tts_data_url = f"data:{mime};base64,{_b64.b64encode(tr.content).decode('ascii')}"
    except httpx.HTTPError as e:
        print(f"❌ tts 에러: {e}")
    return tts_path, tts_data_url


async def _voice_process(nickname: str, audio_path: str, history: list):
    """오디오 파일 → /voice-chat 전사·응답 → 대화기록 + TTS. (history, tts_path, tts_url, ended)."""
    history = history or []
    if nickname not in user_sessions:
        await call_api("/profile", "POST", {"nickname": nickname})
        user_sessions[nickname] = {"started_at": datetime.now().isoformat()}

    headers = BASE_HEADERS
    transcript, response, ended = "", "", False
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT, headers=headers) as client:
            with open(audio_path, "rb") as f:
                files = {"audio": (os.path.basename(audio_path), f, "audio/wav")}
                resp = await client.post(
                    f"{BACKEND_URL}/voice-chat", files=files, data={"nickname": nickname}
                )
            if resp.status_code == 200:
                j = resp.json()
                transcript = (j.get("transcript") or "").strip()
                response = (j.get("response") or "").strip()
                ended = bool(j.get("conversation_ended", False))
            else:
                response = "죄송해요, 음성 처리 중 오류가 났어요. 다시 말씀해 주세요."
    except httpx.HTTPError as e:
        print(f"❌ voice-chat 에러: {type(e).__name__}: {e}")
        response = f"[음성 오류] {type(e).__name__}: {str(e)[:140]}"

    if not transcript:
        history.append({"role": "assistant", "content": response or "잘 못 들었어요. 다시 한 번 말씀해 주시겠어요?"})
        return history, None, "", False

    history.append({"role": "user", "content": f"🎤 {transcript}"})
    history.append({"role": "assistant", "content": response})

    tts_path, tts_data_url = await _synth_tts(response)
    return history, tts_path, tts_data_url, ended


def _ctrl_signal(ended: bool) -> str:
    """핸즈프리 종료 신호 토큰. JS(#hf_ctrl 폴링)가 감지해 마이크를 끈다."""
    import time as _t
    return f"end:{_t.time():.3f}" if ended else ""


async def voice_chat_fn(nickname: str, audio_path, history: list):
    """press-to-talk 마이크(파일경로) 입구. 반환: (history, voice_reply, voice_input, tts_player, hf_ctrl)"""
    if not nickname or not nickname.strip():
        history = history or []
        history.append({"role": "assistant", "content": "닉네임을 먼저 입력하고 시작해 주세요."})
        return history, None, None, gr.update(), gr.update()
    if not audio_path:
        return history or [], None, None, gr.update(), gr.update()
    history, tts_path, tts_url, ended = await _voice_process(nickname, audio_path, history)
    return history, tts_path, None, tts_url, _ctrl_signal(ended)


async def end_conversation_fn(nickname: str, history: list):
    """수동 '대화 종료' 버튼: 백엔드 종료 경로로 마무리 멘트 생성 + TTS. (history, tts_player, hf_ctrl)"""
    history = history or []
    if not nickname or not nickname.strip():
        return history, gr.update(), gr.update()
    res = await call_api("/chat", "POST", {"nickname": nickname, "message": "이제 그만하겠습니다"})
    response = res.get("response") if isinstance(res, dict) else None
    if not response:
        response = "네, 오늘 이야기는 여기까지 하겠습니다. 함께 이야기 나눠서 즐거웠어요. 다음에 또 뵙겠습니다. 건강하세요! 🙏"
    history.append({"role": "assistant", "content": response})
    _tts_path, tts_url = await _synth_tts(response)
    return history, tts_url, _ctrl_signal(True)


_last_b64_hash: Optional[str] = None
_last_b64_at: float = 0.0
_voice_lock: Optional["asyncio.Lock"] = None


async def voice_chat_b64_fn(nickname: str, audio_b64: str, history: list):
    """핸즈프리(브라우저 VAD)가 보낸 base64 WAV 입구."""
    global _last_b64_hash, _last_b64_at, _voice_lock
    import asyncio
    import hashlib
    import time as _time

    # 빈 입력은 출력 리셋으로 인한 spurious 재호출 — 어떤 컴포넌트도 건드리지 않는다.
    # 반환 순서: (history, voice_reply, vad_b64, tts_player, hf_ctrl)
    if not audio_b64:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    # Lock 으로 dedupe 윈도우의 race 를 없앤다 (Gradio queue가 같은 이벤트를 동시에 두 번 보낼 때 보호).
    if _voice_lock is None:
        _voice_lock = asyncio.Lock()
    async with _voice_lock:
        h = hashlib.md5(audio_b64.encode("utf-8")).hexdigest()
        now = _time.time()
        # (a) 같은 b64 가 30초 안에 또 들어오면 같은 발화의 중복 — 무시
        if h == _last_b64_hash and (now - _last_b64_at) < 30:
            print(f"⏭️ 중복 audio_b64 무시 — same hash (h={h[:8]} dt={now - _last_b64_at:.1f}s)")
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        # (b) 직전 처리 후 6초 안에 또 들어오면 같은 이벤트의 즉시 중복발사로 보고 무시.
        #     에코(스피커→마이크) 자체는 재생 중 VAD를 pause()해 클라이언트에서 막으므로,
        #     예전 20초 창은 어르신의 정상적인 다음 발화까지 삼켜 '처리중' 고착을 유발했음 → 축소.
        if (now - _last_b64_at) < 6:
            print(f"⏭️ 6초 내 재진입 무시 — diff hash but too soon (dt={now - _last_b64_at:.1f}s)")
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        _last_b64_hash = h
        _last_b64_at = now

    if not nickname or not nickname.strip():
        history = history or []
        history.append({"role": "assistant", "content": "닉네임을 먼저 입력하고 시작해 주세요."})
        return history, gr.update(), gr.update(), gr.update(), gr.update()
    import base64
    raw = base64.b64decode(audio_b64.split(",")[-1])
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tf.write(raw)
        path = tf.name
    try:
        history, tts_path, tts_url, ended = await _voice_process(nickname, path, history)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    # vad_b64는 gr.update() 로 둔다. tts_player 에 data URL 을 넣어 JS 가 직접 재생.
    # 종료 의도였다면 hf_ctrl 에 신호를 실어 JS 가 마이크를 끄게 한다.
    return history, tts_path, gr.update(), tts_url, _ctrl_signal(ended)


async def on_nickname_submit(nickname: str) -> tuple[str, str]:
    """닉네임 제출 시 인사말 표시"""
    if not nickname.strip():
        return "", "닉네임을 입력해주세요."
    
    greeting = await get_greeting(nickname)
    return greeting, ""


async def get_routine_info(nickname: str) -> str:
    """루틴 정보 조회"""
    if not nickname.strip():
        return "닉네임을 먼저 입력해주세요."
    
    result = await call_api(f"/routine/{nickname}", "GET")
    
    if "error" in result:
        return "루틴 정보를 불러올 수 없습니다."
    
    info_parts = []
    
    current = result.get("current_activity")
    if current and current.get("activity"):
        info_parts.append(f"🕐 **현재**: {current['activity']} ({current.get('minutes_remaining', 0)}분 남음)")
    
    next_act = result.get("next_activity")
    if next_act and next_act.get("activity"):
        info_parts.append(f"⏰ **다음**: {next_act['activity']} ({next_act.get('minutes_until', 0)}분 후)")
    
    summary = result.get("daily_summary", {})
    if summary:
        completed = len(summary.get("completed", []))
        rate = summary.get("completion_rate", 0) * 100
        info_parts.append(f"\n📊 **오늘 완료**: {completed}개 활동 ({rate:.0f}%)")
    
    suggestions = result.get("suggestions", [])
    if suggestions:
        info_parts.append("\n💡 **추천 활동**:")
        for s in suggestions[:3]:
            info_parts.append(f"  • {s}")
    
    return "\n".join(info_parts) if info_parts else "등록된 루틴이 없습니다."


async def save_patient_profile(
    nickname: str,
    name: str,
    age: int,
    conditions: str,
    emergency_contact: str,
    notes: str,
    health_consent: bool = False
) -> str:
    """환자 프로필 저장 (건강정보 동의 여부 반영)"""
    if not nickname.strip():
        return "닉네임을 먼저 입력해주세요."
    
    # 동의하지 않은 경우 건강 관련 정보는 저장하지 않음
    profile_data = {
        "nickname": nickname,
        "name": name or None,
        "age": age if age > 0 else None,
        "emergency_contact": emergency_contact or None,
        "health_info_consent": health_consent,
    }
    
    if health_consent:
        # 동의한 경우에만 건강 상태/질환, 특이사항 저장
        profile_data["conditions"] = conditions or None
        profile_data["notes"] = notes or None
    else:
        # 동의하지 않으면 건강정보 필드를 비움
        profile_data["conditions"] = None
        profile_data["notes"] = None
    
    result = await call_api("/profile", "POST", profile_data)
    
    if "error" in result:
        return f"프로필 저장 실패: {result['error']}"
    
    consent_status = "동의함 ✅" if health_consent else "동의하지 않음"
    return f"✅ {nickname}님의 프로필이 저장되었습니다.\n📋 건강정보 개인화 활용: {consent_status}"


# Gradio UI 테마 및 CSS
CUSTOM_THEME = "soft"

CUSTOM_CSS = """
.container { max-width: 900px; margin: auto; }
.greeting-box { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
.info-box {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #667eea;
}
.status-box {
    padding: 10px 16px;
    border-radius: 20px;
    font-weight: bold;
    text-align: center;
    font-size: 14px;
    background: #f0f0f0;
}
"""

# Gradio UI 구성
# 핸즈프리(브라우저 VAD) 초기화 JS — @ricky0123/vad-web로 무음 감지 후 자동 전송.
# 백엔드는 기존 /voice-chat, /tts 사용. 답변 재생 중에는 VAD를 멈춰 에코를 방지.
HANDSFREE_INIT_JS = """
() => {
  if (window.__hfInit) return;
  window.__hfInit = true;
  window.__hfActive = false;
  window.__hfVAD = null;
  function loadScript(src) {
    return new Promise((resolve, reject) => {
      if (document.querySelector('script[src="' + src + '"]')) { resolve(); return; }
      const s = document.createElement('script');
      s.src = src; s.onload = resolve; s.onerror = reject;
      document.head.appendChild(s);
    });
  }
  function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const w = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
    w(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); w(8, 'WAVE');
    w(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    w(36, 'data'); view.setUint32(40, samples.length * 2, true);
    let off = 44;
    for (let i = 0; i < samples.length; i++, off += 2) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    let bytes = new Uint8Array(buffer), bin = '';
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
  }
  window.__hfStatus = (msg) => {
    const el = document.querySelector('#handsfree_status');
    if (el) { const t = el.querySelector('p, span, .prose'); (t || el).textContent = msg; }
  };
  // 답변 재생 종료 (또는 교체/제거) 시 마이크를 다시 켜고 상태를 정리.
  // 어떤 이벤트(ended/pause/emptied/엘리먼트 제거/safety타이머)가 트리거해도 동일하게 동작한다.
  // VAD 재개는 1.5s 지연 — 스피커 잔향이 마이크로 들어와 self-trigger 되는 것 방지.
  window.__hfResetAfterPlayback = () => {
    if (window.__hfStatusFailsafe) {
      clearTimeout(window.__hfStatusFailsafe);
      window.__hfStatusFailsafe = null;
    }
    // 대화 종료 신호가 있으면 마이크를 재개하지 않고 완전히 끈다.
    if (window.__hfEndRequested) {
      window.__hfEndRequested = false;
      window.__hfActive = false;
      if (window.__hfVAD) { try { window.__hfVAD.pause(); } catch (_) {} }
      window.__hfStatus('✅ 대화가 종료되었습니다. 다시 시작하려면 버튼을 누르세요.');
      window.__hfIgnoreVAD = false;
      window.__hfResetPending = false;
      return;
    }
    if (window.__hfResetPending) return;  // 이미 reset 진행 중이면 중복 안 함
    window.__hfResetPending = true;
    if (!window.__hfActive) {
      window.__hfStatus('⏸️ 중지됨 (버튼을 다시 누르면 시작)');
      window.__hfIgnoreVAD = false;
      window.__hfResetPending = false;
      return;
    }
    window.__hfStatus('🎧 듣는 중… (마이크 안정화)');
    setTimeout(() => {
      if (window.__hfActive && window.__hfVAD) {
        try { window.__hfVAD.start(); } catch (e) {}
        window.__hfStatus('🎧 듣는 중…');
      }
      window.__hfIgnoreVAD = false;
      window.__hfResetPending = false;
    }, 1500);
  };

  // ⭐ 우리가 직접 만든 Audio 객체로 TTS 를 재생한다.
  // Gradio 의 WaveSurfer 플레이어는 네이티브 <audio> 의 src 를 비워둔 채 Web Audio API 로
  // 재생해서 'play'/'ended'/src/currentTime 어느 것도 신뢰할 수 없었음(콘솔 로그로 확인:
  // 'audio 노드 추가 — src=(empty)' 만 찍히고 그 외엔 전무 → 60s 안전타이머만 발동).
  // 그래서 백엔드가 TTS 를 base64 data URL 로 hidden textbox(#tts_player)에 넘기면
  // 그 값을 폴링해 new Audio() 로 직접 재생/종료감지한다. 이 Audio 의 이벤트는 100% 우리 통제.
  window.__hfPlayTTS = (dataUrl) => {
    if (!dataUrl) return;
    if (window.__hfAudio) { try { window.__hfAudio.pause(); } catch (_) {} }
    if (window.__hfStatusFailsafe) { clearTimeout(window.__hfStatusFailsafe); window.__hfStatusFailsafe = null; }
    const a = new Audio(dataUrl);
    window.__hfAudio = a;
    if (window.__hfVAD) { try { window.__hfVAD.pause(); } catch (_) {} }
    window.__hfStatus('🔊 답변 재생 중… (마이크 일시정지)');
    a.addEventListener('ended', () => { console.log('[hf tts] ended — 마이크 재개'); window.__hfResetAfterPlayback(); });
    a.addEventListener('error', () => { console.warn('[hf tts] audio error'); window.__hfResetAfterPlayback(); });
    const p = a.play();
    if (p && p.catch) p.catch(err => {
      console.warn('[hf tts] play() 차단:', err && err.message);
      // 자동재생이 막혀도 마이크는 추정 시간 뒤 풀어줘 영구 stuck 방지.
      const secs = (a.duration && isFinite(a.duration)) ? a.duration : 10;
      window.__hfStatusFailsafe = setTimeout(() => window.__hfResetAfterPlayback(), secs * 1000 + 1000);
    });
  };

  window.__hfHookReply = () => {
    if (window.__hfObsSetup) return;
    window.__hfObsSetup = true;
    // hidden #tts_player textbox 의 값(새 TTS data URL)을 폴링 → 우리 Audio 로 재생.
    setInterval(() => {
      const el = document.querySelector('#tts_player textarea') || document.querySelector('#tts_player input');
      if (!el) return;
      const v = el.value || '';
      if (!v) return;
      const sig = v.length + '|' + v.slice(0, 32);  // 큰 문자열 전체 비교 회피
      if (sig !== window.__hfLastTTSsig) {
        window.__hfLastTTSsig = sig;
        console.log('[hf tts] 새 TTS 감지 — len=' + v.length);
        window.__hfPlayTTS(v);
      }
    }, 300);
    // hidden #hf_ctrl 의 "end:<ts>" 신호를 폴링 → 마무리 멘트 재생 후 마이크 종료.
    setInterval(() => {
      const el = document.querySelector('#hf_ctrl textarea') || document.querySelector('#hf_ctrl input');
      if (!el) return;
      const v = el.value || '';
      if (!v || v === window.__hfLastCtrl) return;
      window.__hfLastCtrl = v;
      if (v.indexOf('end') === 0) {
        console.log('[hf ctrl] 종료 신호 — 마무리 후 마이크 종료');
        window.__hfEndRequested = true;
      }
    }, 300);
  };
  window.hfToggle = async () => {
    try {
      if (window.__hfActive) {
        if (window.__hfVAD) window.__hfVAD.pause();
        window.__hfActive = false; window.__hfStatus('⏸️ 중지됨 (버튼을 다시 누르면 시작)');
        return;
      }
      window.__hfStatus('🎤 마이크 준비 중…');
      if (!window.__hfVADLoaded) {
        await loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js');
        await loadScript('https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.22/dist/bundle.min.js');
        window.__hfVADLoaded = true;
      }
      if (!window.__hfVAD) {
        window.__hfVAD = await vad.MicVAD.new({
          baseAssetPath: 'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.22/dist/',
          onnxWASMBasePath: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/',
          positiveSpeechThreshold: 0.5,
          negativeSpeechThreshold: 0.35,
          minSpeechFrames: 3,
          redemptionFrames: 24,
          onSpeechStart: () => {
            console.log('[VAD] speech start'); window.__hfStatus('🗣️ 말하는 중…');
            // 긴 발화 안정화: 한 발화가 25s 넘게 이어지면 잠깐 쉬어가도록 안내(백엔드는 30s에서 절단).
            if (window.__hfLongSpeechTimer) clearTimeout(window.__hfLongSpeechTimer);
            window.__hfLongSpeechTimer = setTimeout(() => {
              window.__hfStatus('🗣️ 말하는 중… (문장을 마치고 잠깐 쉬시면 더 잘 알아들어요)');
            }, 25000);
          },
          onVADMisfire: () => {
            console.log('[VAD] misfire(너무 짧음)'); window.__hfStatus('🎧 듣는 중… (조금 더 또렷이 말씀해 주세요)');
            if (window.__hfLongSpeechTimer) { clearTimeout(window.__hfLongSpeechTimer); window.__hfLongSpeechTimer = null; }
          },
          onSpeechEnd: (audio) => {
            if (window.__hfLongSpeechTimer) { clearTimeout(window.__hfLongSpeechTimer); window.__hfLongSpeechTimer = null; }
            // 처리/재생 중에는 어떤 VAD 콜백도 무시 — pause()로 라이브러리가 안 멈춰질 때를 대비한 강한 가드.
            if (window.__hfIgnoreVAD) {
              console.log('[VAD] __hfIgnoreVAD 가드 — speech-end 무시');
              return;
            }
            // VAD가 한 발화를 여러 segment로 잘라 onSpeechEnd가 연속 호출되거나
            // 응답 오디오 직후 즉시 트리거되는 경우 — 3초 내 재진입은 무시.
            const _now = Date.now();
            if (_now - (window.__hfLastSpeechEndAt || 0) < 3000) {
              console.log('[VAD] 3초 내 재진입 — speech-end 무시');
              return;
            }
            window.__hfLastSpeechEndAt = _now;
            window.__hfIgnoreVAD = true;  // 다음 reset 까지 무시
            console.log('[VAD] speech end, samples=', audio && audio.length);
            // 발화 종료 즉시 VAD를 멈춘다. 답변 TTS가 마이크로 되돌아 들어와
            // 봇이 자기 목소리에 응답하는 '셀프 에코 루프'를 차단.
            // 응답 오디오 'ended'/'timeupdate'/60s 안전타이머 중 하나가 트리거되면 reset.
            if (window.__hfVAD) { try { window.__hfVAD.pause(); } catch (e) {} }
            window.__hfStatus('⏳ 처리 중…');
            // 마지막 안전망 — 매 발화마다 fresh 60s 타이머. 어떤 이유로든
            // ended/timeupdate/pause 가 트리거되지 않아 상태가 굳으면 자동 풀어준다.
            if (window.__hfStatusFailsafe) clearTimeout(window.__hfStatusFailsafe);
            window.__hfStatusFailsafe = setTimeout(() => {
              console.warn('[hf] 60s 안전 타이머 — 상태 강제 복구');
              window.__hfResetAfterPlayback();
            }, 60000);
            const b64 = encodeWAV(audio, 16000);
            const el = document.querySelector('#vad_b64 textarea') || document.querySelector('#vad_b64 input');
            if (el) {
              const proto = el.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
              Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, b64);
              el.dispatchEvent(new Event('input', { bubbles: true }));
              console.log('[VAD] b64 전송(len=' + b64.length + ')');
            } else { console.warn('[VAD] #vad_b64 입력 요소를 찾지 못함'); }
          }
        });
        console.log('[VAD] MicVAD created');
      }
      await window.__hfVAD.start();
      console.log('[VAD] started, listening');
      window.__hfActive = true; window.__hfStatus('🎧 듣는 중… (말씀하시면 자동 인식)');
      window.__hfHookReply();
    } catch (e) {
      console.error('hands-free error', e);
      window.__hfStatus('❌ 마이크/VAD 초기화 실패 — 브라우저 콘솔을 확인하세요');
    }
  };
}
"""


with gr.Blocks(title="치매노인 맞춤형 헬스케어 챗봇", css="#vad_b64, #tts_player, #hf_ctrl {display: none !important;}") as demo:
    
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown(
                """
                # 🏥 치매노인 맞춤형 헬스케어 챗봇
                
                따뜻하고 친절한 AI 도우미와 대화해보세요. 
                이전 대화를 기억하고 개인화된 케어를 제공합니다.
                """
            )
        with gr.Column(scale=1):
            status_display = gr.Markdown(
                value="🟡 상태 확인 중...",
                elem_classes=["status-box"]
            )
            refresh_status_btn = gr.Button("🔄 새로고침", size="sm")
    
    gr.Markdown(
        """
        > ⏳ **안내**: 서버 절전 모드로 인해 첫 응답에 **1~2분** 정도 소요될 수 있습니다. 
        > 이후 대화는 빠르게 진행됩니다. 잠시만 기다려주세요! 🙏
        """,
        elem_classes=["info-box"]
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            nickname_input = gr.Textbox(
                label="닉네임",
                placeholder="이름 또는 닉네임을 입력하세요",
                info="채팅 시작 전 닉네임을 입력해주세요"
            )
        with gr.Column(scale=1):
            start_btn = gr.Button("시작하기", variant="primary")
    
    greeting_output = gr.Markdown(
        elem_classes=["greeting-box"],
        visible=False
    )
    
    with gr.Tabs() as tabs:
        # 채팅 탭
        with gr.TabItem("💬 대화하기"):
            # Gradio 5.x는 type="messages" 필요, 6.x는 기본값이 messages
            chatbot_kwargs = {"label": "대화", "height": 400}
            if IS_GRADIO_5:
                chatbot_kwargs["type"] = "messages"
            chatbot = gr.Chatbot(**chatbot_kwargs)
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="메시지",
                    placeholder="메시지를 입력하세요...",
                    scale=4
                )
                send_btn = gr.Button("보내기", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("대화 초기화")
                translate_toggle = gr.Checkbox(
                    label="🌐 English Translation (영어 번역)",
                    value=False
                )

            # 음성 대화 (텍스트와 병행)
            with gr.Row():
                handsfree_btn = gr.Button("🎤 음성 대화 시작 / 중지", variant="secondary")
                end_btn = gr.Button("🛑 대화 종료", variant="stop")
                handsfree_status = gr.Markdown("", elem_id="handsfree_status")
            # 백엔드가 직접 서빙하는 저지연(WebSocket 직결+문장 단위 스트리밍) 베타 화면 링크.
            # 새 탭으로 열어 백엔드와 직접 통신하므로 이 Gradio 서버를 거치지 않는다.
            gr.HTML(
                f'<div style="text-align:center;margin:2px 0 8px;">'
                f'<a href="{BACKEND_URL}/ui/" target="_blank" rel="noopener" '
                f'style="font-size:13px;color:#667eea;">⚡ 더 빠른 음성 대화 (베타, 새 창에서 열기)</a></div>'
            )
            with gr.Accordion("🎤 또는 눌러서 말하기 (수동)", open=False):
                voice_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="녹음 후 ■(정지)를 누르면 전송됩니다",
                )
            # autoplay=False — 실제 재생은 JS 가 #tts_player 의 data URL 로 직접 한다.
            # (Gradio WaveSurfer 플레이어는 ended/src 감지가 불가능해 상태 stuck 유발)
            # voice_reply 는 시각적 파형/수동 재생용으로만 유지.
            voice_reply = gr.Audio(
                label="🔊 음성 답변", autoplay=False, interactive=False, elem_id="voice_reply"
            )
            # 핸즈프리 브릿지: visible=False면 DOM에 textarea가 안 생겨 JS가 못 찾으므로,
            # 렌더는 하되 CSS(display:none)로 숨긴다. VAD가 값 주입 → .change로 처리.
            vad_b64 = gr.Textbox(elem_id="vad_b64")
            # TTS data URL 전달용 hidden textbox — JS 가 폴링해 new Audio() 로 재생/종료감지.
            tts_player = gr.Textbox(elem_id="tts_player")
            # 핸즈프리 제어 신호 — "end:<ts>" 가 실리면 JS 가 마이크(VAD)를 끈다.
            hf_ctrl = gr.Textbox(elem_id="hf_ctrl")
        
        # 일과 탭
        with gr.TabItem("📅 일과 관리"):
            routine_output = gr.Markdown(
                elem_classes=["info-box"]
            )
            refresh_routine_btn = gr.Button("새로고침")
        
        # 프로필 탭
        with gr.TabItem("👤 프로필 설정"):
            # IRB 안내문 및 개인정보 동의
            gr.Markdown(
                """
                ### 📋 연구 참여 안내(IRB 승인 연구)
                
                본 챗봇은 **고령자 건강관리 지원을 위한 연구** 목적으로 운영됩니다.
                
                - **연구기관**: (IRB 수정심의 진행 중)
                - **목적**: 고령자 맞춤형 건강 정보 제공 챗봇의 효과성 검증
                - **안내**: 본 챗봇은 연구 목적의 데모용으로 제작되었습니다. **의료적 진단, 약물 투약 관리, 개인화된 의료 처치를 제공하지 않습니다.**
                  건강 관련 전문적인 상담은 반드시 의료 전문가와 상의하시기 바랍니다.
                
                > ⚠️ 아래 "건강정보 개인화 활용 동의"에 체크하시면, 입력하신 건강 상태와 특이사항이
                > 대화 시 맞춤형 응답에 활용됩니다. 동의하지 않으셔도 챗봇 이용에는 제한이 없습니다.
                """,
                elem_classes=["info-box"]
            )
            
            # 건강정보 개인화 활용 동의 체크박스
            health_consent_checkbox = gr.Checkbox(
                label="✅ 건강정보 개인화 활용에 동의합니다(선택사항)",
                value=False,
                info="동의 시 건강 상태/질환, 특이사항이 대화에 반영됩니다. 동의하지 않아도 서비스 이용이 가능합니다."
            )
            
            with gr.Row():
                with gr.Column():
                    profile_name = gr.Textbox(label="이름")
                    profile_age = gr.Number(label="나이", minimum=0, maximum=120)
                    profile_conditions = gr.Textbox(
                        label="건강 상태/질환(동의 시에만 저장됨)",
                        placeholder="예: 고혈압, 당뇨",
                        interactive=False
                    )
                with gr.Column():
                    profile_emergency = gr.Textbox(
                        label="비상 연락처",
                        placeholder="예: 010-1234-5678 (아들)"
                    )
                    profile_notes = gr.Textbox(
                        label="특이사항(동의 시에만 저장됨)",
                        placeholder="예: 아침에 약 드시는 것 잊어버리심",
                        lines=3,
                        interactive=False
                    )
            
            with gr.Row():
                save_profile_btn = gr.Button("프로필 저장", variant="primary")
                delete_history_btn = gr.Button("🗑️ 이전 대화 기록 삭제", variant="stop")
            
            profile_status = gr.Markdown()
            
            # 동의 체크박스 변경 시 건강정보 입력 필드 활성화/비활성화
            def toggle_health_fields(consent):
                return (
                    gr.update(interactive=consent),  # profile_conditions
                    gr.update(interactive=consent),  # profile_notes
                )
            
            health_consent_checkbox.change(
                fn=toggle_health_fields,
                inputs=[health_consent_checkbox],
                outputs=[profile_conditions, profile_notes],
                api_name=False
            )

        # 관리자 탭 (교수/연구자용 — 대화 기록 CSV 다운로드)
        with gr.TabItem("🔐 관리자"):
            gr.Markdown("### 🔐 관리자 메뉴", elem_classes=["info-box"])
            admin_pw = gr.Textbox(
                label="관리자 비밀번호",
                type="password",
                placeholder="비밀번호를 입력하세요"
            )
            admin_login_btn = gr.Button("로그인", variant="primary")
            admin_status = gr.Markdown()

            # 로그인 성공 시에만 노출되는 조회/다운로드 영역
            with gr.Column(visible=False) as admin_panel:
                admin_source = gr.Radio(
                    choices=[
                        "전체 대화 기록 (chat_history)",
                        "분석 로그 — 메타데이터 포함 (conversation_logs)",
                    ],
                    value="전체 대화 기록 (chat_history)",
                    label="데이터 종류"
                )
                with gr.Row():
                    admin_view_btn = gr.Button("👁️ 웹에서 표로 보기", variant="secondary")
                    admin_download_btn = gr.Button("📥 CSV 다운로드", variant="primary")
                admin_result_msg = gr.Markdown()
                admin_table = gr.Dataframe(
                    label="대화 미리보기",
                    interactive=False,
                    wrap=True
                )
                admin_file = gr.File(label="다운로드 파일")

            admin_pw_state = gr.State("")

    # 이벤트 핸들러
    # 상태 변수 (닉네임 잠금 여부)
    nickname_locked = gr.State(False)
    
    async def on_start_or_reset(nickname, is_locked):
        """시작하기/재설정 버튼 클릭 핸들러"""
        if is_locked:
            # 재설정 모드 - 모든 필드 초기화
            return (
                gr.update(visible=False),  # greeting_output
                [],  # chatbot
                gr.update(value="", interactive=True, info="채팅 시작 전 닉네임을 입력해주세요"),  # nickname_input 해제
                gr.update(value="시작하기", variant="primary"),  # start_btn 복원
                False,  # nickname_locked = False
                gr.update(value=""),  # profile_name 초기화
                gr.update(value=0),  # profile_age 초기화
                gr.update(value=""),  # profile_conditions 초기화
                gr.update(value=""),  # profile_emergency 초기화
                gr.update(value=""),  # profile_notes 초기화
                gr.update(value=False),  # health_consent_checkbox 초기화
            )
        else:
            # 시작 모드
            if not nickname.strip():
                return (
                    gr.update(visible=False),
                    [],
                    gr.update(),
                    gr.update(),
                    False,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
            greeting = await get_greeting(nickname)
            # 저장된 프로필 불러오기
            profile = await get_profile(nickname)
            has_consent = profile.get("health_info_consent", False)
            return (
                gr.update(value=greeting, visible=True),  # greeting_output
                [],  # chatbot
                gr.update(interactive=False, info=f"✅ {nickname}님으로 시작됨"),  # nickname_input 잠금
                gr.update(value="🔄 재설정", variant="secondary"),  # start_btn 변경
                True,  # nickname_locked = True
                gr.update(value=profile.get("name", "")),  # profile_name
                gr.update(value=profile.get("age", 0) or 0),  # profile_age
                gr.update(value=profile.get("conditions", ""), interactive=has_consent),  # profile_conditions
                gr.update(value=profile.get("emergency_contact", "")),  # profile_emergency
                gr.update(value=profile.get("notes", ""), interactive=has_consent),  # profile_notes
                gr.update(value=has_consent),  # health_consent_checkbox
            )
    
    async def on_routine_refresh(nickname):
        if not nickname:
            return "닉네임을 먼저 입력해주세요."
        return await get_routine_info(nickname)
    
    start_btn.click(
        fn=on_start_or_reset,
        inputs=[nickname_input, nickname_locked],
        outputs=[greeting_output, chatbot, nickname_input, start_btn, nickname_locked,
                 profile_name, profile_age, profile_conditions, profile_emergency, profile_notes,
                 health_consent_checkbox],
        api_name=False
    )
    
    # 닉네임 입력 후 엔터 → 시작하기 버튼과 동일 동작
    nickname_input.submit(
        fn=on_start_or_reset,
        inputs=[nickname_input, nickname_locked],
        outputs=[greeting_output, chatbot, nickname_input, start_btn, nickname_locked,
                 profile_name, profile_age, profile_conditions, profile_emergency, profile_notes,
                 health_consent_checkbox],
        api_name=False
    )
    
    msg_input.submit(
        fn=chat_with_bot,
        inputs=[nickname_input, msg_input, chatbot, translate_toggle],
        outputs=[chatbot, msg_input],
        api_name=False
    )
    
    send_btn.click(
        fn=chat_with_bot,
        inputs=[nickname_input, msg_input, chatbot, translate_toggle],
        outputs=[chatbot, msg_input],
        api_name=False
    )
    
    clear_btn.click(fn=lambda: [], inputs=None, outputs=chatbot, api_name=False)

    # 음성(수동): 녹음 정지 시 자동 전송 → 전사·응답 표시 + 음성 답변 재생
    voice_input.stop_recording(
        fn=voice_chat_fn,
        inputs=[nickname_input, voice_input, chatbot],
        outputs=[chatbot, voice_reply, voice_input, tts_player, hf_ctrl],
        api_name=False,
    )

    # 핸즈프리: 버튼 클릭 시 브라우저 VAD 토글(JS), 무음 감지되면 hidden 트리거가 처리
    handsfree_btn.click(fn=None, inputs=None, outputs=None, js="() => window.hfToggle()")
    vad_b64.change(
        fn=voice_chat_b64_fn,
        inputs=[nickname_input, vad_b64, chatbot],
        outputs=[chatbot, voice_reply, vad_b64, tts_player, hf_ctrl],
        api_name=False,
    )

    # 대화 종료: 즉시 마이크를 끄고(JS), 백엔드 종료 경로로 마무리 멘트 + TTS 재생
    end_btn.click(
        fn=end_conversation_fn,
        inputs=[nickname_input, chatbot],
        outputs=[chatbot, tts_player, hf_ctrl],
        js="() => { window.__hfEndRequested = true; if (window.__hfActive && window.hfToggle) { window.hfToggle(); } }",
        api_name=False,
    )
    # 페이지 로드 시 핸즈프리 JS 초기화
    demo.load(fn=None, inputs=None, outputs=None, js=HANDSFREE_INIT_JS)
    
    # 상태 체크 이벤트
    refresh_status_btn.click(
        fn=check_backend_status,
        inputs=None,
        outputs=[status_display],
        api_name=False
    )
    
    # 페이지 로드 시 상태 체크
    demo.load(
        fn=check_backend_status,
        inputs=None,
        outputs=[status_display]
    )
    
    refresh_routine_btn.click(
        fn=on_routine_refresh,
        inputs=[nickname_input],
        outputs=[routine_output],
        api_name=False
    )
    
    save_profile_btn.click(
        fn=save_patient_profile,
        inputs=[
            nickname_input,
            profile_name,
            profile_age,
            profile_conditions,
            profile_emergency,
            profile_notes,
            health_consent_checkbox
        ],
        outputs=[profile_status],
        api_name=False
    )
    
    delete_history_btn.click(
        fn=delete_conversation_history,
        inputs=[nickname_input],
        outputs=[profile_status],
        api_name=False
    )

    # 관리자 탭 이벤트
    admin_login_btn.click(
        fn=admin_login,
        inputs=[admin_pw],
        outputs=[admin_status, admin_panel, admin_pw_state],
        api_name=False
    )
    admin_pw.submit(
        fn=admin_login,
        inputs=[admin_pw],
        outputs=[admin_status, admin_panel, admin_pw_state],
        api_name=False
    )
    admin_view_btn.click(
        fn=admin_view_table,
        inputs=[admin_pw_state, admin_source],
        outputs=[admin_table, admin_result_msg],
        api_name=False
    )
    admin_download_btn.click(
        fn=admin_download_csv,
        inputs=[admin_pw_state, admin_source],
        outputs=[admin_file, admin_result_msg],
        api_name=False
    )


# 앱 실행
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
