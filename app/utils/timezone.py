"""
시간대 유틸리티
한국 시간(KST) 처리를 위한 헬퍼 함수들
"""

from datetime import datetime, timezone, timedelta

# 한국 시간대 (KST = UTC+9)
KST = timezone(timedelta(hours=9))


def get_kst_now() -> datetime:
    """현재 한국 시간 반환"""
    return datetime.now(KST)


def get_kst_hour() -> int:
    """현재 한국 시간의 시(hour) 반환"""
    return get_kst_now().hour


def get_kst_date_str() -> str:
    """현재 한국 날짜 문자열 반환 (YYYY-MM-DD)"""
    return get_kst_now().strftime("%Y-%m-%d")


def get_kst_datetime_str() -> str:
    """현재 한국 날짜시간 문자열 반환"""
    return get_kst_now().strftime("%Y-%m-%d %H:%M:%S KST")


def format_kst_time(dt: datetime) -> str:
    """datetime을 한국 시간 문자열로 변환"""
    if dt.tzinfo is None:
        # naive datetime이면 KST로 간주
        dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST).strftime("%H:%M")
