"""
Gradio 프론트엔드 애플리케이션
Hugging Face Spaces 배포용
"""

import os
import httpx
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
IS_HUGGINGFACE = os.getenv("SPACE_ID") is not None

print(f"📦 Gradio 버전: {gr.__version__} (5.x: {IS_GRADIO_5}, HF Spaces: {IS_HUGGINGFACE})")

# 환경변수에서 백엔드 URL 가져오기
# HuggingFace Spaces면 Cloud Run URL 사용, 로컬이면 localhost
if IS_HUGGINGFACE:
    BACKEND_URL = os.getenv("BACKEND_URL", "https://acronymous-nonobsessive-chong.ngrok-free.dev")
else:
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# 상태 저장용
user_sessions = {}

print(f"🔗 Backend URL: {BACKEND_URL}")

# Cloud Run 콜드 스타트 + LLM 응답 시간 고려 (최대 180초)
API_TIMEOUT = 180.0


async def call_api(endpoint: str, method: str = "GET", data: Optional[dict] = None) -> dict:
    """백엔드 API 호출"""
    # ngrok 무료 플랜: 브라우저 경고 페이지 우회 헤더
    headers = {"ngrok-skip-browser-warning": "true"}
    async with httpx.AsyncClient(timeout=API_TIMEOUT, headers=headers) as client:
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
with gr.Blocks(title="치매노인 맞춤형 헬스케어 챗봇") as demo:
    
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


# 앱 실행
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
