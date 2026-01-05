"""
스케줄러 모듈 (선택적)
정기적인 작업 실행 (복약 알림, 루틴 체크 등)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Callable, Optional
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class TaskScheduler:
    """간단한 태스크 스케줄러"""
    
    def __init__(self):
        self._tasks: list[dict] = []
        self._running = False
    
    def add_task(
        self,
        name: str,
        func: Callable,
        interval_minutes: int,
        args: tuple = (),
        kwargs: dict = None
    ):
        """
        태스크 추가
        
        Args:
            name: 태스크 이름
            func: 실행할 함수
            interval_minutes: 실행 간격 (분)
            args: 함수 인자
            kwargs: 함수 키워드 인자
        """
        self._tasks.append({
            "name": name,
            "func": func,
            "interval": interval_minutes,
            "args": args,
            "kwargs": kwargs or {},
            "last_run": None,
            "next_run": datetime.now()
        })
        logger.info(f"태스크 추가됨: {name} (간격: {interval_minutes}분)")
    
    async def run_task(self, task: dict):
        """단일 태스크 실행"""
        try:
            func = task["func"]
            if asyncio.iscoroutinefunction(func):
                await func(*task["args"], **task["kwargs"])
            else:
                func(*task["args"], **task["kwargs"])
            
            task["last_run"] = datetime.now()
            task["next_run"] = datetime.now() + timedelta(minutes=task["interval"])
            logger.debug(f"태스크 완료: {task['name']}")
        
        except Exception as e:
            logger.error(f"태스크 오류 ({task['name']}): {e}")
    
    async def start(self):
        """스케줄러 시작"""
        self._running = True
        logger.info("스케줄러 시작됨")
        
        while self._running:
            now = datetime.now()
            
            for task in self._tasks:
                if task["next_run"] <= now:
                    await self.run_task(task)
            
            await asyncio.sleep(60)  # 1분마다 체크
    
    def stop(self):
        """스케줄러 중지"""
        self._running = False
        logger.info("스케줄러 중지됨")


# 스케줄러 인스턴스
scheduler = TaskScheduler()


async def check_medication_reminders():
    """복약 알림 체크 태스크"""
    from app.healthcare import MedicationReminder
    from app.vector_store import get_chroma_handler
    
    chroma = get_chroma_handler()
    reminder = MedicationReminder()
    
    # 모든 활성 환자 조회 (간단한 구현)
    # 실제로는 별도 환자 목록 관리 필요
    logger.debug("복약 알림 체크 중...")


async def check_daily_routines():
    """일일 루틴 체크 태스크"""
    from app.healthcare import DailyRoutineManager
    
    manager = DailyRoutineManager()
    logger.debug("일일 루틴 체크 중...")


def setup_scheduler():
    """스케줄러 설정"""
    if settings.MEDICATION_REMINDER_ENABLED:
        scheduler.add_task(
            name="복약 알림 체크",
            func=check_medication_reminders,
            interval_minutes=15
        )
    
    if settings.DAILY_ROUTINE_TRACKING:
        scheduler.add_task(
            name="일일 루틴 체크",
            func=check_daily_routines,
            interval_minutes=30
        )
    
    return scheduler
