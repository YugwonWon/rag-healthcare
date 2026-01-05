"""
임베딩 모델 테스트
"""

import pytest
import numpy as np


def test_embedding_model_load():
    """임베딩 모델 로드 테스트"""
    from app.model.local_model import get_embedding_model
    
    model = get_embedding_model()
    assert model is not None
    assert model.dimension == 384


def test_embedding_single_text():
    """단일 텍스트 임베딩 테스트"""
    from app.model.local_model import get_embedding_model
    
    model = get_embedding_model()
    embedding = model.embed_query("안녕하세요, 오늘 기분이 어떠세요?")
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)


def test_embedding_multiple_texts():
    """다중 텍스트 임베딩 테스트"""
    from app.model.local_model import get_embedding_model
    
    model = get_embedding_model()
    texts = [
        "약 먹을 시간입니다.",
        "오늘 산책 가실까요?",
        "머리가 아파요."
    ]
    
    embeddings = model.embed_documents(texts)
    
    assert len(embeddings) == 3
    assert all(len(emb) == 384 for emb in embeddings)


def test_embedding_similarity():
    """임베딩 유사도 테스트"""
    from app.model.local_model import get_embedding_model
    
    model = get_embedding_model()
    
    # 유사한 문장
    emb1 = np.array(model.embed_query("약 먹을 시간이에요"))
    emb2 = np.array(model.embed_query("약 복용 시간입니다"))
    
    # 다른 문장
    emb3 = np.array(model.embed_query("오늘 날씨가 좋아요"))
    
    # 코사인 유사도 계산
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim_similar = cosine_similarity(emb1, emb2)
    sim_different = cosine_similarity(emb1, emb3)
    
    # 유사한 문장은 더 높은 유사도를 가져야 함
    assert sim_similar > sim_different


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
