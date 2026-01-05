"""
쿼리 핸들러 테스트
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_chroma_handler():
    """ChromaDB 핸들러 모킹"""
    with patch("app.retriever.query_handler.get_chroma_handler") as mock:
        handler = MagicMock()
        handler.get_patient_profile.return_value = {
            "nickname": "테스트",
            "name": "김테스트",
            "age": 80,
            "conditions": "치매"
        }
        handler.search_documents.return_value = {
            "documents": [["치매 환자 케어 가이드라인"]],
            "metadatas": [[{"source": "healthcare_docs"}]]
        }
        handler.get_user_conversations.return_value = {
            "documents": [["이전 대화 내용"]],
            "metadatas": [[{"timestamp": "2025-01-03T10:00:00"}]]
        }
        handler.add_conversation.return_value = "conv_123"
        handler.get_recent_activities.return_value = [
            {
                "timestamp": "2025-01-03T10:00:00",
                "message": "산책 다녀올게요",
                "document": ""
            }
        ]
        mock.return_value = handler
        yield handler


@pytest.fixture
def mock_llm():
    """LLM 모킹"""
    with patch("app.retriever.query_handler.get_llm") as mock:
        llm = MagicMock()
        llm.chat = AsyncMock(return_value="안녕하세요, 테스트님! 오늘도 좋은 하루 보내세요.")
        mock.return_value = llm
        yield llm


@pytest.mark.asyncio
async def test_process_query(mock_chroma_handler, mock_llm):
    """쿼리 처리 테스트"""
    from app.retriever.query_handler import RAGQueryHandler
    
    handler = RAGQueryHandler()
    response = await handler.process_query(
        nickname="테스트",
        query="안녕하세요"
    )
    
    assert response is not None
    assert isinstance(response, str)
    mock_chroma_handler.add_conversation.assert_called_once()


@pytest.mark.asyncio
async def test_generate_greeting(mock_chroma_handler, mock_llm):
    """인사말 생성 테스트"""
    from app.retriever.query_handler import RAGQueryHandler
    from datetime import datetime
    
    handler = RAGQueryHandler()
    greeting = await handler.generate_greeting("테스트")
    
    assert "테스트" in greeting
    mock_chroma_handler.get_recent_activities.assert_called()


def test_format_patient_info():
    """환자 정보 포맷팅 테스트"""
    from app.retriever.query_handler import RAGQueryHandler
    
    handler = RAGQueryHandler()
    
    # 프로필 있는 경우
    profile = {"name": "김테스트", "age": 80, "nickname": "테스트"}
    result = handler._format_patient_info(profile)
    assert "name" in result
    assert "age" in result
    
    # 프로필 없는 경우
    result = handler._format_patient_info(None)
    assert "등록된 환자 정보가 없습니다" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
