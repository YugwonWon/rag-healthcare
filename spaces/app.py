"""
Gradio 프론트엔드 애플리케이션
Hugging Face Spaces 배포용
"""

import os
import httpx
import gradio as gr
from datetime import datetime
from typing import Optional

# 환경변수에서 백엔드 URL 가져오기
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")

# 상태 저장용
user_sessions = {}


async def call_api(endpoint: str, method: str = "GET", data: Optional[dict] = None) -> dict:
    """백엔드 API 호출"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        url = f"{BACKEND_URL}{endpoint}"
        
        try:
            if method == "GET":
                response = await client.get(url, params=data)
            else:
                response = await client.post(url, json=data)
            
            response.raise_for_status()
            return response.json()
        
        except httpx.HTTPError as e:
            return {"error": str(e)}


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


async def chat_with_bot(
    nickname: str,
    message: str,
    history: list,
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
    notes: str
) -> str:
    """환자 프로필 저장"""
    if not nickname.strip():
        return "닉네임을 먼저 입력해주세요."
    
    result = await call_api("/profile", "POST", {
        "nickname": nickname,
        "name": name or None,
        "age": age if age > 0 else None,
        "conditions": conditions or None,
        "emergency_contact": emergency_contact or None,
        "notes": notes or None
    })
    
    if "error" in result:
        return f"프로필 저장 실패: {result['error']}"
    
    return f"✅ {nickname}님의 프로필이 저장되었습니다."


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
"""

# Gradio UI 구성
with gr.Blocks(title="치매노인 맞춤형 헬스케어 챗봇") as demo:
    
    gr.Markdown(
        """
        # 🏥 치매노인 맞춤형 헬스케어 챗봇
        
        따뜻하고 친절한 AI 도우미와 대화해보세요. 
        이전 대화를 기억하고 개인화된 케어를 제공합니다.
        """
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
            chatbot = gr.Chatbot(
                label="대화",
                height=400,
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="메시지",
                    placeholder="메시지를 입력하세요...",
                    scale=4
                )
                send_btn = gr.Button("보내기", variant="primary", scale=1)
            
            clear_btn = gr.Button("대화 초기화")
        
        # 일과 탭
        with gr.TabItem("📅 일과 관리"):
            routine_output = gr.Markdown(
                elem_classes=["info-box"]
            )
            refresh_routine_btn = gr.Button("새로고침")
        
        # 프로필 탭
        with gr.TabItem("👤 프로필 설정"):
            with gr.Row():
                with gr.Column():
                    profile_name = gr.Textbox(label="이름")
                    profile_age = gr.Number(label="나이", minimum=0, maximum=120)
                    profile_conditions = gr.Textbox(
                        label="건강 상태/질환",
                        placeholder="예: 고혈압, 당뇨"
                    )
                with gr.Column():
                    profile_emergency = gr.Textbox(
                        label="비상 연락처",
                        placeholder="예: 010-1234-5678 (아들)"
                    )
                    profile_notes = gr.Textbox(
                        label="특이사항",
                        placeholder="예: 아침에 약 드시는 것 잊어버리심",
                        lines=3
                    )
            
            save_profile_btn = gr.Button("프로필 저장", variant="primary")
            profile_status = gr.Markdown()
    
    # 이벤트 핸들러
    async def on_start(nickname):
        if not nickname.strip():
            return gr.update(visible=False), []
        greeting = await get_greeting(nickname)
        return gr.update(value=greeting, visible=True), []
    
    start_btn.click(
        on_start,
        inputs=[nickname_input],
        outputs=[greeting_output, chatbot]
    )
    
    msg_input.submit(
        chat_with_bot,
        inputs=[nickname_input, msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )
    
    send_btn.click(
        chat_with_bot,
        inputs=[nickname_input, msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )
    
    clear_btn.click(lambda: [], None, chatbot)
    
    refresh_routine_btn.click(
        get_routine_info,
        inputs=[nickname_input],
        outputs=[routine_output]
    )
    
    save_profile_btn.click(
        save_patient_profile,
        inputs=[
            nickname_input,
            profile_name,
            profile_age,
            profile_conditions,
            profile_emergency,
            profile_notes
        ],
        outputs=[profile_status]
    )
    
    # 탭 변경 시 루틴 정보 로드
    tabs.select(
        lambda nickname: get_routine_info(nickname) if nickname else "닉네임을 먼저 입력해주세요.",
        inputs=[nickname_input],
        outputs=[routine_output]
    )


# 앱 실행
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=CUSTOM_THEME,
        css=CUSTOM_CSS
    )
