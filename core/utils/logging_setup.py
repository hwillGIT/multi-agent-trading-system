"""
Logging configuration for the trading system.
"""

import sys
from pathlib import Path
from loguru import logger

from ..base.config import config


def setup_logging():
    """
    Configure logging for the trading system using loguru.
    """
    # Remove default handler
    logger.remove()
    
    # Get logging configuration
    log_level = config.get("logging.level", "INFO")
    log_format = config.get("logging.format", 
                           "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}")
    log_file = config.get("logging.file_path", "logs/trading_system.log")
    max_file_size = config.get("logging.max_file_size", "100MB")
    backup_count = config.get("logging.backup_count", 5)
    audit_log_path = config.get("logging.audit_log_path", "logs/audit.log")
    
    # Ensure log directory exists
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    audit_dir = Path(audit_log_path).parent
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    # Console handler
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File handler for general logs
    logger.add(
        log_file,
        format=log_format,
        level=log_level,
        rotation=max_file_size,
        retention=backup_count,
        backtrace=True,
        diagnose=True,
        encoding="utf-8"
    )
    
    # Separate audit log for compliance
    logger.add(
        audit_log_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | AUDIT | {extra[agent]} | {message}",
        level="INFO",
        rotation="1 day",
        retention="30 days",
        filter=lambda record: "audit" in record["extra"],
        encoding="utf-8"
    )
    
    logger.info("Logging system initialized")


def get_audit_logger():
    """Get a logger instance for audit trails."""
    return logger.bind(audit=True)