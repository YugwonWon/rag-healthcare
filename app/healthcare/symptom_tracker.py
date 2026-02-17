"""
증상 추적 모듈
치매노인의 증상 및 상태 변화 기록
"""

from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

# TODO: pgvector 마이그레이션 시 LangChainStore 연동
from app.utils import get_kst_now


class SymptomSeverity(Enum):
    """증상 심각도"""
    MILD = "경미"
    MODERATE = "보통"
    SEVERE = "심각"
    CRITICAL = "위급"


class MoodType(Enum):
    """기분 유형"""
    HAPPY = "좋음"
    NEUTRAL = "보통"
    ANXIOUS = "불안"
    SAD = "우울"
    CONFUSED = "혼란"
    AGITATED = "초조"


@dataclass
class SymptomRecord:
    """증상 기록"""
    symptom_type: str
    severity: SymptomSeverity
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    notes: Optional[str] = None


@dataclass
class MoodRecord:
    """기분 기록"""
    mood: MoodType
    energy_level: int  # 1-10
    timestamp: datetime = field(default_factory=datetime.now)
    notes: Optional[str] = None


class SymptomTracker:
    """증상 추적기"""
    
    # 주요 증상 키워드 매핑
    SYMPTOM_KEYWORDS = {
        "두통": ["머리", "아프", "두통", "지끈"],
        "어지러움": ["어지러", "빙빙", "균형"],
        "기억력": ["기억", "잊어", "생각안나", "까먹"],
        "수면": ["잠", "못자", "불면", "깨"],
        "식욕": ["밥", "못먹", "입맛", "식욕"],
        "통증": ["아프", "쑤시", "통증", "뻐근"],
        "호흡": ["숨", "답답", "가슴"],
        "배변": ["변비", "설사", "화장실"],
    }
    
    # 위험 키워드
    ALERT_KEYWORDS = ["쓰러", "넘어", "피", "의식", "숨", "응급", "119", "병원"]
    
    def __init__(self):
        # in-memory 저장 (TODO: pgvector 마이그레이션)
        self._symptom_logs: list[dict] = []
        self._mood_logs: list[dict] = []
    
    def analyze_message(self, nickname: str, message: str) -> dict:
        """
        메시지에서 증상 분석
        
        Args:
            nickname: 사용자 닉네임
            message: 사용자 메시지
        
        Returns:
            분석 결과 딕셔너리
        """
        detected_symptoms = []
        alert_level = "normal"
        recommendations = []
        
        message_lower = message.lower()
        
        # 위험 키워드 체크
        for keyword in self.ALERT_KEYWORDS:
            if keyword in message_lower:
                alert_level = "critical"
                recommendations.append("⚠️ 보호자 또는 의료진에게 즉시 연락이 필요합니다.")
                break
        
        # 증상 키워드 체크
        for symptom, keywords in self.SYMPTOM_KEYWORDS.items():
            for keyword in keywords:
                if keyword in message_lower:
                    detected_symptoms.append(symptom)
                    break
        
        # 증상별 권장사항
        if "두통" in detected_symptoms:
            recommendations.append("충분히 쉬시고, 물을 드세요. 두통이 계속되면 보호자에게 알려주세요.")
        if "어지러움" in detected_symptoms:
            recommendations.append("천천히 움직이시고, 앉아서 쉬세요. 혼자 걷지 마세요.")
        if "기억력" in detected_symptoms:
            recommendations.append("걱정하지 마세요. 중요한 것은 메모해 두시면 좋아요.")
        if "수면" in detected_symptoms:
            recommendations.append("규칙적인 수면 시간을 유지하고, 자기 전 따뜻한 우유가 도움이 됩니다.")
        
        # 증상 기록 저장
        if detected_symptoms:
            self._save_symptom_record(nickname, detected_symptoms, message)
        
        return {
            "detected_symptoms": detected_symptoms,
            "alert_level": alert_level,
            "recommendations": recommendations,
            "needs_attention": len(detected_symptoms) > 2 or alert_level == "critical"
        }
    
    def record_symptom(
        self,
        nickname: str,
        symptom_type: str,
        severity: SymptomSeverity,
        description: str,
        notes: Optional[str] = None
    ) -> SymptomRecord:
        """
        증상 기록 저장
        
        Args:
            nickname: 환자 닉네임
            symptom_type: 증상 유형
            severity: 심각도
            description: 설명
            notes: 추가 메모
        
        Returns:
            증상 기록 객체
        """
        record = SymptomRecord(
            symptom_type=symptom_type,
            severity=severity,
            description=description,
            notes=notes
        )
        
        self._save_symptom_record(
            nickname,
            [symptom_type],
            f"{symptom_type}: {description} (심각도: {severity.value})"
        )
        
        return record
    
    def record_mood(
        self,
        nickname: str,
        mood: MoodType,
        energy_level: int,
        notes: Optional[str] = None
    ) -> MoodRecord:
        """
        기분 기록 저장
        
        Args:
            nickname: 환자 닉네임
            mood: 기분 유형
            energy_level: 에너지 레벨 (1-10)
            notes: 추가 메모
        
        Returns:
            기분 기록 객체
        """
        record = MoodRecord(
            mood=mood,
            energy_level=min(max(energy_level, 1), 10),
            notes=notes
        )
        
        mood_text = f"기분: {mood.value}, 에너지: {energy_level}/10"
        if notes:
            mood_text += f", 메모: {notes}"
        
        self._mood_logs.append({
            "nickname": nickname,
            "type": "mood_record",
            "mood": mood.value,
            "energy_level": energy_level,
            "text": mood_text,
            "timestamp": get_kst_now().isoformat()
        })
        
        return record
    
    def get_symptom_history(
        self,
        nickname: str,
        days: int = 7
    ) -> list[dict]:
        """
        증상 기록 히스토리 조회
        
        Args:
            nickname: 환자 닉네임
            days: 조회 기간 (일)
        
        Returns:
            증상 기록 리스트
        """
        # in-memory에서 해당 사용자 기록 조회
        symptom_records = [
            log for log in self._symptom_logs + self._mood_logs
            if log.get("nickname") == nickname
        ]
        return symptom_records[-50:]  # 최근 50건
    
    def _save_symptom_record(
        self,
        nickname: str,
        symptoms: list[str],
        message: str
    ) -> None:
        """증상 기록 저장 (내부 메서드)"""
        self._symptom_logs.append({
            "nickname": nickname,
            "type": "symptom_record",
            "symptoms": ",".join(symptoms),
            "message": message,
            "timestamp": get_kst_now().isoformat()
        })
