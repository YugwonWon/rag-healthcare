"""
일상 루틴 관리 모듈
치매노인의 일상 활동 추적 및 관리
"""

from datetime import datetime, time, timedelta
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from app.config import settings
from app.vector_store import get_chroma_handler
from app.utils import get_kst_now, KST


class ActivityType(Enum):
    """활동 유형"""
    WAKE_UP = "기상"
    BREAKFAST = "아침식사"
    LUNCH = "점심식사"
    DINNER = "저녁식사"
    SNACK = "간식"
    MEDICATION = "복약"
    EXERCISE = "운동"
    WALK = "산책"
    NAP = "낮잠"
    TV = "TV시청"
    READING = "독서"
    HOBBY = "취미활동"
    SOCIAL = "사회활동"
    BATH = "목욕/씻기"
    SLEEP = "취침"
    OTHER = "기타"


class CompletionStatus(Enum):
    """완료 상태"""
    PENDING = "예정"
    COMPLETED = "완료"
    SKIPPED = "건너뜀"
    PARTIALLY = "일부완료"


@dataclass
class RoutineItem:
    """루틴 항목"""
    activity_type: ActivityType
    scheduled_time: time
    duration_minutes: int = 30
    description: Optional[str] = None
    is_required: bool = True  # 필수 활동 여부
    reminder_before_minutes: int = 10


@dataclass
class ActivityLog:
    """활동 기록"""
    activity_type: ActivityType
    scheduled_time: datetime
    actual_time: Optional[datetime] = None
    status: CompletionStatus = CompletionStatus.PENDING
    notes: Optional[str] = None
    mood_after: Optional[str] = None  # 활동 후 기분


class DailyRoutineManager:
    """일상 루틴 관리자"""
    
    # 기본 루틴 템플릿
    DEFAULT_ROUTINE = [
        RoutineItem(ActivityType.WAKE_UP, time(7, 0), 30, "기상 및 세면"),
        RoutineItem(ActivityType.BREAKFAST, time(8, 0), 45, "아침 식사"),
        RoutineItem(ActivityType.MEDICATION, time(8, 30), 5, "아침 약 복용"),
        RoutineItem(ActivityType.EXERCISE, time(9, 30), 30, "가벼운 스트레칭"),
        RoutineItem(ActivityType.WALK, time(10, 30), 30, "산책"),
        RoutineItem(ActivityType.LUNCH, time(12, 0), 45, "점심 식사"),
        RoutineItem(ActivityType.MEDICATION, time(12, 30), 5, "점심 약 복용"),
        RoutineItem(ActivityType.NAP, time(14, 0), 60, "낮잠", is_required=False),
        RoutineItem(ActivityType.HOBBY, time(15, 30), 60, "취미 활동", is_required=False),
        RoutineItem(ActivityType.DINNER, time(18, 0), 45, "저녁 식사"),
        RoutineItem(ActivityType.MEDICATION, time(18, 30), 5, "저녁 약 복용"),
        RoutineItem(ActivityType.TV, time(19, 30), 60, "TV 시청", is_required=False),
        RoutineItem(ActivityType.BATH, time(20, 30), 30, "목욕/씻기"),
        RoutineItem(ActivityType.SLEEP, time(21, 30), 0, "취침"),
    ]
    
    def __init__(self):
        self._chroma = get_chroma_handler()
        self._routines: dict[str, list[RoutineItem]] = {}
        self._activity_logs: dict[str, list[ActivityLog]] = {}
    
    def initialize_routine(
        self,
        nickname: str,
        custom_routine: Optional[list[RoutineItem]] = None
    ) -> list[RoutineItem]:
        """
        루틴 초기화
        
        Args:
            nickname: 환자 닉네임
            custom_routine: 커스텀 루틴 (없으면 기본 루틴 사용)
        
        Returns:
            설정된 루틴 리스트
        """
        routine = custom_routine or self.DEFAULT_ROUTINE.copy()
        self._routines[nickname] = routine
        
        # 프로필에 저장
        self._save_routine_to_profile(nickname, routine)
        
        return routine
    
    def get_routine(self, nickname: str) -> list[RoutineItem]:
        """
        루틴 조회
        
        Args:
            nickname: 환자 닉네임
        
        Returns:
            루틴 리스트
        """
        if nickname not in self._routines:
            self.initialize_routine(nickname)
        return self._routines.get(nickname, [])
    
    def get_current_activity(self, nickname: str) -> Optional[dict]:
        """
        현재 시간의 활동 조회
        
        Args:
            nickname: 환자 닉네임
        
        Returns:
            현재 활동 정보 또는 None
        """
        now = get_kst_now()
        current_time = now.time()
        routine = self.get_routine(nickname)
        
        for i, item in enumerate(routine):
            scheduled_time = item.scheduled_time
            end_time = (
                datetime.combine(now.date(), scheduled_time, tzinfo=KST) + 
                timedelta(minutes=item.duration_minutes)
            ).time()
            
            if scheduled_time <= current_time <= end_time:
                return {
                    "item": item,
                    "is_current": True,
                    "minutes_remaining": self._minutes_until(end_time)
                }
        
        return None
    
    def get_next_activity(self, nickname: str) -> Optional[dict]:
        """
        다음 활동 조회
        
        Args:
            nickname: 환자 닉네임
        
        Returns:
            다음 활동 정보 또는 None
        """
        now = get_kst_now()
        current_time = now.time()
        routine = self.get_routine(nickname)
        
        for item in routine:
            if item.scheduled_time > current_time:
                return {
                    "item": item,
                    "minutes_until": self._minutes_until(item.scheduled_time)
                }
        
        # 내일 첫 활동
        if routine:
            return {
                "item": routine[0],
                "is_tomorrow": True
            }
        
        return None
    
    def record_activity(
        self,
        nickname: str,
        activity_type: ActivityType,
        status: CompletionStatus = CompletionStatus.COMPLETED,
        notes: Optional[str] = None,
        mood_after: Optional[str] = None
    ) -> ActivityLog:
        """
        활동 기록
        
        Args:
            nickname: 환자 닉네임
            activity_type: 활동 유형
            status: 완료 상태
            notes: 메모
            mood_after: 활동 후 기분
        
        Returns:
            활동 기록 객체
        """
        now = get_kst_now()
        
        log = ActivityLog(
            activity_type=activity_type,
            scheduled_time=now,
            actual_time=now,
            status=status,
            notes=notes,
            mood_after=mood_after
        )
        
        if nickname not in self._activity_logs:
            self._activity_logs[nickname] = []
        self._activity_logs[nickname].append(log)
        
        # ChromaDB에 저장
        self._chroma.add_conversation(
            nickname=nickname,
            user_message=f"{activity_type.value} {status.value}",
            assistant_response=f"✅ {activity_type.value} 기록됨 ({now.strftime('%H:%M')})",
            metadata={
                "type": "activity_log",
                "activity_type": activity_type.value,
                "status": status.value,
                "notes": notes,
                "mood_after": mood_after
            }
        )
        
        return log
    
    def get_daily_summary(self, nickname: str, date: Optional[datetime] = None) -> dict:
        """
        일일 활동 요약
        
        Args:
            nickname: 환자 닉네임
            date: 조회 날짜 (기본: 오늘)
        
        Returns:
            일일 요약 딕셔너리
        """
        target_date = (date or get_kst_now()).strftime("%Y-%m-%d")
        
        results = self._chroma.get_user_conversations(
            nickname=nickname,
            query="활동",
            n_results=50
        )
        
        completed = []
        skipped = []
        pending = []
        
        if results and results.get("metadatas"):
            metadatas = results.get("metadatas", [])
            if isinstance(metadatas[0], list):
                metadatas = metadatas[0]
            
            for metadata in metadatas:
                if metadata.get("type") == "activity_log":
                    if metadata.get("date", "") == target_date:
                        status = metadata.get("status", "")
                        activity = metadata.get("activity_type", "")
                        
                        if status == CompletionStatus.COMPLETED.value:
                            completed.append(activity)
                        elif status == CompletionStatus.SKIPPED.value:
                            skipped.append(activity)
                        else:
                            pending.append(activity)
        
        routine = self.get_routine(nickname)
        total_required = sum(1 for r in routine if r.is_required)
        
        return {
            "date": target_date,
            "completed": completed,
            "skipped": skipped,
            "pending": pending,
            "completion_rate": len(completed) / total_required if total_required > 0 else 0,
            "total_activities": len(completed) + len(skipped)
        }
    
    def generate_routine_message(self, nickname: str) -> str:
        """
        현재 루틴 상태 메시지 생성
        
        Args:
            nickname: 환자 닉네임
        
        Returns:
            루틴 상태 메시지
        """
        current = self.get_current_activity(nickname)
        next_activity = self.get_next_activity(nickname)
        
        message_parts = []
        
        if current:
            item = current["item"]
            remaining = current.get("minutes_remaining", 0)
            message_parts.append(
                f"🕐 지금은 {item.activity_type.value} 시간이에요. "
                f"({remaining}분 남음)"
            )
        
        if next_activity and not next_activity.get("is_tomorrow"):
            item = next_activity["item"]
            minutes = next_activity.get("minutes_until", 0)
            message_parts.append(
                f"⏰ 다음은 {item.activity_type.value}이에요. "
                f"({minutes}분 후)"
            )
        
        if not message_parts:
            message_parts.append("오늘 하루도 잘 마무리하셨네요! 😊")
        
        return "\n".join(message_parts)
    
    def _minutes_until(self, target_time: time) -> int:
        """지정 시간까지 남은 분 계산"""
        now = get_kst_now()
        target = datetime.combine(now.date(), target_time, tzinfo=KST)
        
        if target < now:
            target += timedelta(days=1)
        
        return int((target - now).total_seconds() / 60)
    
    def _save_routine_to_profile(
        self,
        nickname: str,
        routine: list[RoutineItem]
    ) -> None:
        """루틴을 프로필에 저장"""
        profile = self._chroma.get_patient_profile(nickname) or {}
        
        routine_summary = ", ".join([
            f"{r.activity_type.value}({r.scheduled_time.strftime('%H:%M')})"
            for r in routine[:5]  # 처음 5개만 저장
        ])
        
        profile["routine_summary"] = routine_summary
        profile["has_routine"] = "yes"
        self._chroma.save_patient_profile(nickname, profile)
    
    def get_activity_suggestions(self, nickname: str) -> list[str]:
        """
        활동 제안 생성
        
        Args:
            nickname: 환자 닉네임
        
        Returns:
            활동 제안 리스트
        """
        now = get_kst_now()
        hour = now.hour
        
        suggestions = []
        
        if 6 <= hour < 10:
            suggestions = [
                "창문을 열어 신선한 공기를 마셔보세요 🌅",
                "가벼운 스트레칭으로 몸을 풀어보세요",
                "물 한 잔 마시는 것도 좋아요 💧"
            ]
        elif 10 <= hour < 12:
            suggestions = [
                "날씨가 좋으면 잠깐 산책 어떠세요? 🚶",
                "좋아하는 음악을 들어보세요 🎵",
                "가족에게 전화해보는 건 어때요? 📞"
            ]
        elif 12 <= hour < 14:
            suggestions = [
                "맛있는 점심 드셨나요? 🍽️",
                "식후에 잠깐 쉬는 것도 좋아요",
            ]
        elif 14 <= hour < 17:
            suggestions = [
                "좋아하는 TV 프로그램 시청은 어떠세요? 📺",
                "간단한 퍼즐이나 게임도 좋아요 🧩",
                "따뜻한 차 한 잔 어떠세요? ☕"
            ]
        elif 17 <= hour < 20:
            suggestions = [
                "저녁 식사 준비 시간이에요 🍽️",
                "하루를 돌아보며 일기를 써보세요 📝",
            ]
        else:
            suggestions = [
                "편안한 음악과 함께 휴식하세요 🎶",
                "잠들기 전 따뜻한 물을 드세요",
                "오늘 하루도 수고하셨어요 💤"
            ]
        
        return suggestions
