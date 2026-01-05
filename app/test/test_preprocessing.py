"""
NER + N-gram 전처리 모듈 테스트
"""

import pytest
from unittest.mock import patch, MagicMock


class TestKoreanNERProcessor:
    """한국어 NER 프로세서 테스트"""
    
    def test_health_keywords_direct_matching(self):
        """키워드 직접 매칭 테스트 (NER 모델 없이)"""
        from app.preprocessing.korean_ner import KoreanNERProcessor
        
        processor = KoreanNERProcessor()
        
        # 테스트 케이스들
        test_cases = [
            ("오늘 병원에 다녀왔어요", ["병원"]),
            ("어제 잠을 못 잤어요", ["잠"]),
            ("두통이 심해요", ["두통"]),
            ("약을 먹었어요", ["약"]),
            ("산책하고 밥 먹었어요", ["산책", "밥"]),
            ("안녕하세요", []),  # 건강 관련 키워드 없음
        ]
        
        for text, expected_keywords in test_cases:
            result = processor.process(text, use_model=False)
            found_terms = [e.text for e in result.health_entities]
            
            for keyword in expected_keywords:
                assert any(keyword in term for term in found_terms), \
                    f"'{text}'에서 '{keyword}'를 찾지 못함: {found_terms}"
    
    def test_overlapping_entities_removal(self):
        """겹치는 개체 제거 테스트"""
        from app.preprocessing.korean_ner import KoreanNERProcessor, NamedEntity
        
        processor = KoreanNERProcessor()
        
        entities = [
            NamedEntity("수면", "HEALTH", 0, 2, 1.0),
            NamedEntity("수면제", "HEALTH", 0, 3, 1.0),  # 더 긴 것
        ]
        
        result = processor._remove_overlapping_entities(entities)
        assert len(result) == 1
        assert result[0].text == "수면제"


class TestNGramExtractor:
    """N-gram 추출기 테스트"""
    
    def test_tokenize_korean(self):
        """한국어 토큰화 테스트"""
        from app.preprocessing.ngram_extractor import NGramExtractor
        
        extractor = NGramExtractor()
        
        text = "오늘 병원에 다녀왔어요"
        tokens = extractor.tokenize(text)
        
        assert len(tokens) > 0
        assert all(len(t) == 3 for t in tokens)  # (토큰, 시작, 끝)
    
    def test_extract_context(self):
        """컨텍스트 추출 테스트"""
        from app.preprocessing.ngram_extractor import NGramExtractor
        
        extractor = NGramExtractor(n_before=3, n_after=3)
        
        text = "어제 아침에 병원에 가서 진료를 받고 약을 처방받았어요"
        # "병원" 위치: 6-8
        
        context = extractor.extract_context(text, "병원", 6, 8)
        
        assert context.target_term == "병원"
        assert len(context.before_words) <= 3
        assert len(context.after_words) <= 3
        assert context.full_context != ""
    
    def test_extract_all_contexts(self):
        """다중 용어 컨텍스트 추출 테스트"""
        from app.preprocessing.ngram_extractor import NGramExtractor
        
        extractor = NGramExtractor()
        
        text = "병원에서 약 처방받고 집에서 잠을 잤어요"
        health_terms = [
            ("병원", 0, 2),
            ("약", 5, 6),
            ("잠", 18, 19),
        ]
        
        result = extractor.extract_all_contexts(text, health_terms)
        
        assert len(result.contexts) == 3
        assert result.combined_context != ""


class TestHealthSignalDetector:
    """건강 위험 신호 감지기 테스트"""
    
    def test_detect_risk_by_keywords(self):
        """키워드 기반 위험 감지 테스트"""
        from app.preprocessing.health_signal_detector import (
            HealthSignalDetector, RiskCategory, RiskLevel
        )
        
        detector = HealthSignalDetector(use_ner_model=False)
        
        test_cases = [
            ("어제 넘어져서 다쳤어요", RiskCategory.FALL_RISK),
            ("잠을 못 자서 피곤해요", RiskCategory.SLEEP_DISORDER),
            ("두통이 심해요", RiskCategory.PAIN),
            ("밥을 못 먹겠어요", RiskCategory.NUTRITIONAL),
            ("요즘 많이 우울해요", RiskCategory.EMOTIONAL),
            ("자꾸 깜빡해요", RiskCategory.COGNITIVE),
        ]
        
        for text, expected_category in test_cases:
            result = detector.analyze(text)
            categories = [s.category for s in result.risk_signals]
            
            assert expected_category in categories, \
                f"'{text}'에서 {expected_category}를 감지하지 못함: {categories}"
    
    def test_overall_risk_level(self):
        """전체 위험 수준 계산 테스트"""
        from app.preprocessing.health_signal_detector import (
            HealthSignalDetector, RiskLevel
        )
        
        detector = HealthSignalDetector(use_ner_model=False)
        
        # 낙상 (HIGH) + 통증 (MEDIUM) = HIGH
        result = detector.analyze("넘어져서 무릎이 아파요")
        assert result.overall_risk_level in [RiskLevel.HIGH, RiskLevel.MEDIUM]
        
        # 응급 (CRITICAL)
        result = detector.analyze("가슴이 아프고 숨을 못 쉬겠어요")
        assert result.overall_risk_level == RiskLevel.CRITICAL
    
    def test_enhanced_query_generation(self):
        """향상된 쿼리 생성 테스트"""
        from app.preprocessing.health_signal_detector import HealthSignalDetector
        
        detector = HealthSignalDetector(use_ner_model=False)
        
        result = detector.analyze("요즘 잠을 못 자요")
        
        assert result.enhanced_query != ""
        assert "잠" in result.enhanced_query or "수면" in result.enhanced_query
    
    def test_summary_generation(self):
        """요약 생성 테스트"""
        from app.preprocessing.health_signal_detector import HealthSignalDetector
        
        detector = HealthSignalDetector(use_ner_model=False)
        
        result = detector.analyze("두통이 심하고 어지러워요")
        
        assert result.summary != ""
        assert "위험 수준" in result.summary


class TestIntegration:
    """통합 테스트"""
    
    def test_preprocess_for_rag(self):
        """RAG 전처리 함수 테스트"""
        from app.preprocessing.health_signal_detector import preprocess_for_rag
        
        text = "병원에서 약 처방받았는데 두통이 안 나아요"
        enhanced = preprocess_for_rag(text, use_ner_model=False)
        
        assert len(enhanced) >= len(text)
    
    def test_get_risk_summary(self):
        """API용 요약 함수 테스트"""
        from app.preprocessing.health_signal_detector import HealthSignalDetector
        
        detector = HealthSignalDetector(use_ner_model=False)
        
        summary = detector.get_risk_summary("낙상 후 무릎 통증이 있어요")
        
        assert "overall_risk" in summary
        assert "detected_health_terms" in summary
        assert "risk_categories" in summary
        assert "enhanced_query" in summary


if __name__ == "__main__":
    # 간단한 수동 테스트
    print("=== NER + N-gram 전처리 테스트 ===\n")
    
    from app.preprocessing.korean_ner import KoreanNERProcessor
    from app.preprocessing.ngram_extractor import NGramExtractor
    from app.preprocessing.health_signal_detector import HealthSignalDetector
    
    # NER 테스트 (모델 없이)
    print("1. NER 테스트 (키워드 매칭)")
    ner = KoreanNERProcessor()
    test_text = "어제 병원에서 진료 받았는데 두통약을 처방받았어요. 잠도 잘 못 자요."
    result = ner.process(test_text, use_model=False)
    print(f"   텍스트: {test_text}")
    print(f"   건강 용어: {[e.text for e in result.health_entities]}\n")
    
    # N-gram 테스트
    print("2. N-gram 컨텍스트 추출")
    ngram = NGramExtractor(n_before=3, n_after=3)
    health_terms = [(e.text, e.start, e.end) for e in result.health_entities]
    ngram_result = ngram.extract_all_contexts(test_text, health_terms)
    for ctx in ngram_result.contexts:
        print(f"   타겟: {ctx.target_term}")
        print(f"   앞: {ctx.before_words}, 뒤: {ctx.after_words}")
        print(f"   컨텍스트: {ctx.full_context}\n")
    
    # 건강 위험 감지
    print("3. 건강 위험 신호 감지")
    detector = HealthSignalDetector(use_ner_model=False)
    analysis = detector.analyze("요즘 자주 넘어지고 어지러워요. 밥도 잘 못 먹겠어요.")
    print(f"   전체 위험 수준: {analysis.overall_risk_level.value}")
    for signal in analysis.risk_signals:
        print(f"   - {signal.category.value}: {signal.level.value}")
        print(f"     설명: {signal.description}")
    print(f"   향상된 쿼리: {analysis.enhanced_query[:100]}...")
    print()
    
    print("✅ 모든 테스트 완료!")
