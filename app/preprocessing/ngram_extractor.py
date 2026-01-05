"""
N-gram 추출 모듈
논문 방법론에 따라 태깅된 건강 용어 전후 N개 단어를 추출
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class NGramContext:
    """N-gram 컨텍스트 정보"""
    target_term: str  # 태깅된 건강 용어
    target_start: int  # 원문에서 시작 위치
    target_end: int  # 원문에서 끝 위치
    before_words: List[str]  # 앞 N개 단어
    after_words: List[str]  # 뒤 N개 단어
    before_text: str  # 앞 컨텍스트 원문
    after_text: str  # 뒤 컨텍스트 원문
    full_context: str  # 전체 컨텍스트 (앞 + 타겟 + 뒤)


@dataclass
class NGramResult:
    """N-gram 추출 결과"""
    original_text: str
    contexts: List[NGramContext] = field(default_factory=list)
    combined_context: str = ""  # 모든 컨텍스트 병합


class NGramExtractor:
    """N-gram 추출기
    
    논문 방법론에 따라 건강 관련 태깅 용어 전후 N개 단어를 추출합니다.
    
    "N-grams consisting of five words before and after each tagged term 
    are combined with semantic similarity analysis"
    """
    
    # 한국어 토큰화 패턴 (공백 + 조사/어미 분리)
    KOREAN_TOKEN_PATTERN = re.compile(
        r'[가-힣]+(?:[은는이가을를에서로와과의도만])?|'  # 한글 + 조사
        r'[a-zA-Z]+|'  # 영문
        r'\d+(?:\.\d+)?|'  # 숫자
        r'[^\s\w]'  # 특수문자
    )
    
    # 의미없는 단어 (stop words)
    STOP_WORDS = {
        "은", "는", "이", "가", "을", "를", "에", "의", "도", "만",
        "그", "저", "이것", "그것", "저것", "여기", "거기", "저기",
        "것", "수", "등", "및", "또", "또는", "그리고", "하지만",
        "으로", "에서", "로", "와", "과", "요", "네", "예", "아니요"
    }
    
    def __init__(
        self,
        n_before: int = 5,
        n_after: int = 5,
        min_context_words: int = 2,
        include_target_in_context: bool = True
    ):
        """
        Args:
            n_before: 타겟 용어 앞 추출할 단어 수 (기본 5)
            n_after: 타겟 용어 뒤 추출할 단어 수 (기본 5)
            min_context_words: 최소 컨텍스트 단어 수
            include_target_in_context: 전체 컨텍스트에 타겟 포함 여부
        """
        self.n_before = n_before
        self.n_after = n_after
        self.min_context_words = min_context_words
        self.include_target_in_context = include_target_in_context
        
    def tokenize(self, text: str) -> List[Tuple[str, int, int]]:
        """한국어 텍스트 토큰화
        
        Returns:
            List[(토큰, 시작위치, 끝위치)]
        """
        tokens = []
        for match in self.KOREAN_TOKEN_PATTERN.finditer(text):
            token = match.group()
            if token.strip():  # 빈 토큰 제외
                tokens.append((token, match.start(), match.end()))
        return tokens
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """간단한 공백 기반 토큰화 (역호환용)"""
        # 공백으로 분리 후 빈 문자열 제거
        tokens = text.split()
        return [t for t in tokens if t.strip()]
    
    def _get_word_at_position(
        self, 
        tokens: List[Tuple[str, int, int]], 
        position: int,
        direction: str = "before"
    ) -> List[str]:
        """특정 위치 기준으로 N개 단어 추출
        
        Args:
            tokens: 토큰 리스트 [(토큰, 시작, 끝)]
            position: 기준 위치
            direction: "before" 또는 "after"
            
        Returns:
            추출된 단어 리스트
        """
        if direction == "before":
            # 위치 이전의 토큰 찾기
            before_tokens = [t for t in tokens if t[2] <= position]
            # 마지막 N개 선택
            selected = before_tokens[-self.n_before:] if before_tokens else []
            return [t[0] for t in selected]
        else:
            # 위치 이후의 토큰 찾기
            after_tokens = [t for t in tokens if t[1] >= position]
            # 처음 N개 선택
            selected = after_tokens[:self.n_after] if after_tokens else []
            return [t[0] for t in selected]
    
    def extract_context(
        self, 
        text: str, 
        target_term: str, 
        target_start: int, 
        target_end: int
    ) -> NGramContext:
        """특정 타겟 용어의 N-gram 컨텍스트 추출
        
        Args:
            text: 전체 텍스트
            target_term: 타겟 건강 용어
            target_start: 타겟 시작 위치
            target_end: 타겟 끝 위치
            
        Returns:
            NGramContext: 추출된 컨텍스트
        """
        # 토큰화
        tokens = self.tokenize(text)
        
        # 앞/뒤 단어 추출
        before_words = self._get_word_at_position(tokens, target_start, "before")
        after_words = self._get_word_at_position(tokens, target_end, "after")
        
        # stop words 필터링 (선택적)
        before_words_filtered = [w for w in before_words if w not in self.STOP_WORDS]
        after_words_filtered = [w for w in after_words if w not in self.STOP_WORDS]
        
        # 필터링 후에도 최소 단어 수 유지
        if len(before_words_filtered) < self.min_context_words:
            before_words_filtered = before_words
        if len(after_words_filtered) < self.min_context_words:
            after_words_filtered = after_words
        
        # 텍스트 형태로 변환
        before_text = " ".join(before_words_filtered)
        after_text = " ".join(after_words_filtered)
        
        # 전체 컨텍스트 구성
        if self.include_target_in_context:
            full_context = f"{before_text} {target_term} {after_text}".strip()
        else:
            full_context = f"{before_text} {after_text}".strip()
        
        return NGramContext(
            target_term=target_term,
            target_start=target_start,
            target_end=target_end,
            before_words=before_words_filtered,
            after_words=after_words_filtered,
            before_text=before_text,
            after_text=after_text,
            full_context=full_context
        )
    
    def extract_all_contexts(
        self, 
        text: str, 
        health_terms: List[Tuple[str, int, int]]
    ) -> NGramResult:
        """모든 건강 용어에 대해 N-gram 컨텍스트 추출
        
        Args:
            text: 전체 텍스트
            health_terms: 건강 용어 리스트 [(용어, 시작, 끝)]
            
        Returns:
            NGramResult: 모든 컨텍스트 결과
        """
        result = NGramResult(original_text=text)
        
        for term, start, end in health_terms:
            context = self.extract_context(text, term, start, end)
            result.contexts.append(context)
        
        # 모든 컨텍스트 병합 (중복 제거)
        all_words = set()
        for ctx in result.contexts:
            all_words.update(ctx.before_words)
            all_words.add(ctx.target_term)
            all_words.update(ctx.after_words)
        
        result.combined_context = " ".join(sorted(all_words, key=lambda x: text.find(x) if x in text else 0))
        
        return result
    
    def get_context_for_embedding(
        self, 
        text: str, 
        health_terms: List[Tuple[str, int, int]]
    ) -> str:
        """임베딩용 컨텍스트 문자열 생성
        
        논문의 semantic similarity 분석을 위한 텍스트 생성
        
        Returns:
            임베딩에 사용할 컨텍스트 문자열
        """
        if not health_terms:
            # 건강 용어가 없으면 원문 그대로 반환
            return text
            
        result = self.extract_all_contexts(text, health_terms)
        
        # 각 컨텍스트의 full_context를 연결
        contexts = [ctx.full_context for ctx in result.contexts if ctx.full_context]
        
        if not contexts:
            return text
            
        # 중복 제거하면서 순서 유지
        seen = set()
        unique_contexts = []
        for ctx in contexts:
            if ctx not in seen:
                seen.add(ctx)
                unique_contexts.append(ctx)
        
        return " ".join(unique_contexts)


def extract_ngram_context(
    text: str,
    health_terms: List[Tuple[str, int, int]],
    n_before: int = 5,
    n_after: int = 5
) -> str:
    """간편한 N-gram 컨텍스트 추출 함수
    
    Args:
        text: 원문
        health_terms: NER로 추출된 건강 용어 [(용어, 시작, 끝)]
        n_before: 앞 단어 수
        n_after: 뒤 단어 수
        
    Returns:
        임베딩용 컨텍스트 문자열
    """
    extractor = NGramExtractor(n_before=n_before, n_after=n_after)
    return extractor.get_context_for_embedding(text, health_terms)
