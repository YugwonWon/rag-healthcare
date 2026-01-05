"""
로깅 설정 모듈
프로젝트 전체 로깅 설정 및 자식 로거 관리
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Optional

# 로그 설정 상수
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = "healthcare_rag.log"
MAX_BYTES = 50 * 1024 * 1024  # 50MB
BACKUP_COUNT = 5  # 최대 5개 파일 유지

# 로그 포맷
DETAILED_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-25s | "
    "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
)
SIMPLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# 날짜 포맷
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_level: str = "INFO",
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """
    프로젝트 전체 로깅 설정
    
    Args:
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_console: 콘솔 출력 여부
        log_to_file: 파일 저장 여부
        log_dir: 로그 디렉토리 (기본: ./logs)
    
    Returns:
        루트 로거
    """
    # 로그 디렉토리 생성
    log_directory = log_dir or LOG_DIR
    log_directory.mkdir(parents=True, exist_ok=True)
    
    # 루트 로거 설정
    root_logger = logging.getLogger("healthcare_rag")
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # 기존 핸들러 제거 (중복 방지)
    root_logger.handlers.clear()
    
    # 콘솔 핸들러
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            ColoredFormatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT)
        )
        root_logger.addHandler(console_handler)
    
    # 파일 핸들러 (RotatingFileHandler)
    if log_to_file:
        log_file_path = log_directory / LOG_FILE
        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(DETAILED_FORMAT, datefmt=DATE_FORMAT)
        )
        root_logger.addHandler(file_handler)
        
        # 에러 전용 파일 핸들러
        error_log_path = log_directory / "healthcare_rag_error.log"
        error_handler = RotatingFileHandler(
            filename=error_log_path,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter(DETAILED_FORMAT, datefmt=DATE_FORMAT)
        )
        root_logger.addHandler(error_handler)
    
    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    return root_logger


class ColoredFormatter(logging.Formatter):
    """콘솔용 컬러 포맷터"""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    """
    모듈별 자식 로거 가져오기
    
    Args:
        name: 로거 이름 (보통 __name__ 사용)
    
    Returns:
        자식 로거
    
    Example:
        from app.logger import get_logger
        logger = get_logger(__name__)
        logger.info("메시지")
    """
    # healthcare_rag 하위 로거로 생성
    if name.startswith("app."):
        # app.module.submodule -> healthcare_rag.module.submodule
        child_name = name.replace("app.", "healthcare_rag.", 1)
    else:
        child_name = f"healthcare_rag.{name}"
    
    return logging.getLogger(child_name)


def log_function_call(logger: logging.Logger):
    """함수 호출 로깅 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"→ {func.__name__}() 호출 | args={args[:3]}... kwargs={list(kwargs.keys())}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"← {func.__name__}() 완료")
                return result
            except Exception as e:
                logger.error(f"✗ {func.__name__}() 실패: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


def log_startup_info(logger: logging.Logger, app_name: str, version: str, config: dict):
    """서비스 시작 정보 로깅"""
    logger.info("=" * 60)
    logger.info(f"🚀 {app_name} v{version} 시작")
    logger.info("=" * 60)
    logger.info(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python 버전: {sys.version}")
    logger.info(f"로그 디렉토리: {LOG_DIR}")
    logger.info("-" * 60)
    logger.info("설정 정보:")
    for key, value in config.items():
        # 민감 정보 마스킹
        if "key" in key.lower() or "secret" in key.lower() or "password" in key.lower():
            value = "***MASKED***"
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)


def log_request(logger: logging.Logger, method: str, path: str, nickname: Optional[str] = None):
    """API 요청 로깅"""
    user_info = f" | user={nickname}" if nickname else ""
    logger.info(f"📥 {method} {path}{user_info}")


def log_response(logger: logging.Logger, method: str, path: str, status_code: int, duration_ms: float):
    """API 응답 로깅"""
    status_emoji = "✅" if status_code < 400 else "❌"
    logger.info(f"📤 {status_emoji} {method} {path} | status={status_code} | {duration_ms:.2f}ms")


# 모듈 임포트 시 기본 로깅 설정 (환경변수 기반)
_root_logger: Optional[logging.Logger] = None


def init_logging() -> logging.Logger:
    """로깅 초기화 (앱 시작 시 한 번만 호출)"""
    global _root_logger
    if _root_logger is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
        log_to_console = os.getenv("LOG_TO_CONSOLE", "true").lower() == "true"
        log_to_file = os.getenv("LOG_TO_FILE", "true").lower() == "true"
        
        _root_logger = setup_logging(
            log_level=log_level,
            log_to_console=log_to_console,
            log_to_file=log_to_file,
        )
    return _root_logger
