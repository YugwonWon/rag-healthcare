"""
복약 알림 모듈
치매노인을 위한 복약 스케줄 관리
"""

from datetime import datetime, time, timedelta
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
import json

from app.config import settings, prompts
from app.utils import get_kst_now, KST


class MedicationFrequency(Enum):
    """복약 빈도"""
    ONCE_DAILY = "1일 1회"
    TWICE_DAILY = "1일 2회"
    THREE_TIMES_DAILY = "1일 3회"
    AS_NEEDED = "필요시"
    WEEKLY = "주 1회"


class MealRelation(Enum):
    """식사 관련"""
    BEFORE_MEAL = "식전"
    AFTER_MEAL = "식후"
    WITH_MEAL = "식사와 함께"
    EMPTY_STOMACH = "공복"
    ANYTIME = "상관없음"


@dataclass
class Medication:
    """복약 정보"""
    name: str
    dosage: str
    frequency: MedicationFrequency
    times: list[time]  # 복용 시간들
    meal_relation: MealRelation
    notes: Optional[str] = None
    start_date: datetime = field(default_factory=datetime.now)
    end_date: Optional[datetime] = None
    is_active: bool = True


@dataclass
class MedicationLog:
    """복약 기록"""
    medication_name: str
    scheduled_time: datetime
    taken_time: Optional[datetime] = None
    was_taken: bool = False
    notes: Optional[str] = None


class MedicationReminder:
    """복약 알림 관리자"""
    
    def __init__(self):
        self._medications: dict[str, list[Medication]] = {}  # nickname -> medications
    
    def add_medication(
        self,
        nickname: str,
        name: str,
        dosage: str,
        frequency: MedicationFrequency,
        times: list[str],  # ["08:00", "20:00"] 형식
        meal_relation: MealRelation = MealRelation.AFTER_MEAL,
        notes: Optional[str] = None
    ) -> Medication:
        """
        복약 정보 추가
        
        Args:
            nickname: 환자 닉네임
            name: 약 이름
            dosage: 복용량
            frequency: 복용 빈도
            times: 복용 시간 리스트 (HH:MM 형식)
            meal_relation: 식사 관련
            notes: 추가 메모
        
        Returns:
            생성된 Medication 객체
        """
        parsed_times = []
        for t in times:
            hour, minute = map(int, t.split(":"))
            parsed_times.append(time(hour, minute))
        
        medication = Medication(
            name=name,
            dosage=dosage,
            frequency=frequency,
            times=parsed_times,
            meal_relation=meal_relation,
            notes=notes
        )
        
        if nickname not in self._medications:
            self._medications[nickname] = []
        self._medications[nickname].append(medication)
        
        # 프로필에 저장
        self._save_medication_to_profile(nickname, medication)
        
        return medication
    
    def get_due_medications(
        self,
        nickname: str,
        within_minutes: int = 30
    ) -> list[dict]:
        """
        복용 예정 약 조회
        
        Args:
            nickname: 환자 닉네임
            within_minutes: 확인할 시간 범위 (분)
        
        Returns:
            복용 예정 약 리스트
        """
        now = get_kst_now()
        current_time = now.time()
        
        due_medications = []
        
        medications = self._medications.get(nickname, [])
        for med in medications:
            if not med.is_active:
                continue
            
            for med_time in med.times:
                # 시간 차이 계산 (timezone-aware)
                med_datetime = datetime.combine(now.date(), med_time, tzinfo=KST)
                time_diff = (med_datetime - now).total_seconds() / 60
                
                if -5 <= time_diff <= within_minutes:  # 5분 전 ~ within_minutes분 후
                    due_medications.append({
                        "medication": med,
                        "scheduled_time": med_time,
                        "is_overdue": time_diff < 0
                    })
        
        return due_medications
    
    def generate_reminder_message(
        self,
        nickname: str,
        medication: Medication
    ) -> str:
        """
        복약 알림 메시지 생성
        
        Args:
            nickname: 환자 닉네임
            medication: 약 정보
        
        Returns:
            알림 메시지
        """
        return prompts.MEDICATION_REMINDER.format(
            nickname=nickname,
            medication_name=medication.name,
            dosage=medication.dosage
        )
    
    def record_medication_taken(
        self,
        nickname: str,
        medication_name: str,
        notes: Optional[str] = None
    ) -> MedicationLog:
        """
        복약 완료 기록
        
        Args:
            nickname: 환자 닉네임
            medication_name: 약 이름
            notes: 메모
        
        Returns:
            복약 기록 객체
        """
        now = get_kst_now()
        
        log = MedicationLog(
            medication_name=medication_name,
            scheduled_time=now,  # 실제로는 예정 시간 찾아야 함
            taken_time=now,
            was_taken=True,
            notes=notes
        )
        
        # 기록 저장 (인메모리)
        # TODO: pgvector 스토어로 이전 가능
        
        return log
    
    def get_medication_history(
        self,
        nickname: str,
        days: int = 7
    ) -> list[dict]:
        """
        복약 기록 조회
        
        Args:
            nickname: 환자 닉네임
            days: 조회 기간
        
        Returns:
            복약 기록 리스트
        """
        # TODO: pgvector 스토어로 이전 가능
        medication_logs = []
        return medication_logs
    
    def get_adherence_rate(
        self,
        nickname: str,
        days: int = 7
    ) -> float:
        """
        복약 준수율 계산
        
        Args:
            nickname: 환자 닉네임
            days: 계산 기간
        
        Returns:
            준수율 (0.0 ~ 1.0)
        """
        logs = self.get_medication_history(nickname, days)
        
        if not logs:
            return 0.0
        
        taken_count = sum(1 for log in logs if log.get("was_taken", False))
        return taken_count / len(logs)
    
    def _save_medication_to_profile(
        self,
        nickname: str,
        medication: Medication
    ) -> None:
        """복약 정보를 프로필에 저장"""
        # TODO: pgvector 스토어로 이전 가능
        pass
    
    def check_and_send_reminders(self, nickname: str) -> list[str]:
        """
        알림 확인 및 메시지 생성
        
        Args:
            nickname: 환자 닉네임
        
        Returns:
            알림 메시지 리스트
        """
        if not settings.MEDICATION_REMINDER_ENABLED:
            return []
        
        due_meds = self.get_due_medications(nickname)
        reminders = []
        
        for item in due_meds:
            med = item["medication"]
            is_overdue = item["is_overdue"]
            
            if is_overdue:
                msg = f"⏰ {nickname}님, {med.name} 드실 시간이 조금 지났어요. 지금이라도 드세요!"
            else:
                msg = self.generate_reminder_message(nickname, med)
            
            reminders.append(msg)
        
        return reminders
