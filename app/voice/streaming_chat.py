"""문장 단위 스트리밍 음성 대화 오케스트레이터.

기존 LangGraph 컴파일 그래프(app/graph/graph.py)는 건드리지 않고 우회한다.
preprocess_node/classify_intent_node/retrieve_node를 직접 호출해 동일한
전처리·검색 로직을 재사용하고, generate_response_node의 LLM 호출 부분만
스트리밍으로 바꿔 문장이 끝나는 즉시 TTS까지 합성해 yield한다.

응급/차단/종료처럼 안전이 중요하거나 응답이 고정된 인텐트는 기존 전체 그래프
(RAGQueryHandler.process_query)로 그대로 위임한다 — 응급 로직을 새로 만들지 않음.
"""

import asyncio
import re
from typing import AsyncIterator, Optional

from app.graph import ConversationState, Intent
from app.graph.nodes import (
    build_system_prompt,
    classify_intent_node,
    preprocess_node,
    retrieve_node,
    save_conversation_node,
)
from app.model import get_llm
from app.logger import get_logger

logger = get_logger(__name__)

# 안전·고정 응답 인텐트는 기존 전체 그래프로 위임 (그래프가 자체 저장하므로
# stream_voice_turn에서 또 저장하지 않는다 — 중복 저장 방지).
_FALLBACK_INTENTS = (Intent.BLOCKED, Intent.EMERGENCY, Intent.END_CONVERSATION)

# 문장 경계: 마침표/물음표/느낌표/줄임표 + 공백(또는 끝). 소수점(예: "3.5")은
# 앞이 숫자면 매칭하지 않아 분리하지 않는다.
_SENTENCE_BOUNDARY = re.compile(r"(?<!\d)([.!?…])(?=\s|$)")
# 문장부호가 한참 안 나오는 경우 다음 공백에서 강제로 잘라 첫 음성 지연을 제한한다.
_FORCE_FLUSH_CHARS = 80


async def _sentence_stream(token_iter: AsyncIterator[str]) -> AsyncIterator[str]:
    """토큰 델타 스트림을 받아 완성된 문장 단위로 yield한다."""
    buffer = ""
    async for delta in token_iter:
        buffer += delta
        while True:
            m = _SENTENCE_BOUNDARY.search(buffer)
            if m:
                cut = m.end()
                sentence, buffer = buffer[:cut].strip(), buffer[cut:]
                if sentence:
                    yield sentence
                continue
            if len(buffer) >= _FORCE_FLUSH_CHARS:
                sp = buffer.rfind(" ")
                if sp >= _FORCE_FLUSH_CHARS // 2:
                    sentence, buffer = buffer[:sp].strip(), buffer[sp:]
                    if sentence:
                        yield sentence
                    continue
            break
    tail = buffer.strip()
    if tail:
        yield tail


async def stream_voice_turn(nickname: str, message: str) -> AsyncIterator[dict]:
    """한 턴을 처리하며 문장이 완성되는 즉시 텍스트+음성 청크를 yield한다.

    yield되는 dict 형태:
        {"type": "text_chunk", "text": str, "audio": Optional[bytes]}
        {"type": "done", "conversation_ended": bool}
    """
    from app.voice import tts as voice_tts  # 지연 임포트 (순환 임포트 방지)
    from app.model.local_model import OllamaClient

    state: ConversationState = {"nickname": nickname, "message": message}
    state.update(await preprocess_node(state))
    state.update(classify_intent_node(state))

    intent = state.get("intent", Intent.GENERAL_CHAT)

    if intent in _FALLBACK_INTENTS:
        from app.retriever.query_handler import get_query_handler
        result = await get_query_handler().process_query(nickname, message)
        response = result.get("response", "")
        if response:
            audio = await _synthesize_safe(voice_tts, response)
            yield {"type": "text_chunk", "text": response, "audio": audio}
        yield {"type": "done", "conversation_ended": result.get("conversation_ended", False)}
        return

    if intent != Intent.GENERAL_CHAT:
        state.update(retrieve_node(state))

    system_prompt, medical_referral_given = build_system_prompt(state)
    state["system_prompt"] = system_prompt
    state["medical_referral_given"] = medical_referral_given

    llm = get_llm()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]

    full_response_parts: list[str] = []
    if hasattr(llm, "chat_stream"):
        token_iter = llm.chat_stream(messages)
        async for sentence in _sentence_stream(token_iter):
            sentence = OllamaClient._postprocess_exaone(sentence)
            if not sentence:
                continue
            full_response_parts.append(sentence)
            audio = await _synthesize_safe(voice_tts, sentence)
            yield {"type": "text_chunk", "text": sentence, "audio": audio}
    else:
        # OpenAIModel 등 스트리밍 미지원 백엔드 폴백 — 한 번에 받아 단일 청크로.
        response = await llm.chat(messages)
        if response:
            full_response_parts.append(response)
            audio = await _synthesize_safe(voice_tts, response)
            yield {"type": "text_chunk", "text": response, "audio": audio}

    full_response = " ".join(full_response_parts).strip()
    state["response"] = full_response
    logger.info(f"💬 스트리밍 응답 완료 | intent={intent.value} | len={len(full_response)}")

    await save_conversation_node(state)

    yield {"type": "done", "conversation_ended": False}


async def _synthesize_safe(voice_tts_module, text: str) -> Optional[bytes]:
    """TTS 합성 실패해도 텍스트 청크는 보낼 수 있도록 예외를 흡수한다.

    synthesize()는 동기(blocking) HTTP 호출이라 그대로 await하면 이벤트 루프를
    막아 동시 접속자 전원의 WS ping이 끊긴다(부하 시 연결 끊김의 주원인).
    asyncio.to_thread로 워커 스레드에 떠넘겨 루프를 계속 응답 가능하게 둔다.
    """
    try:
        audio, _media_type = await asyncio.to_thread(voice_tts_module.synthesize, text)
        return audio
    except Exception as e:  # noqa: BLE001
        logger.error(f"문장 TTS 합성 오류(텍스트만 전달): {e}")
        return None
