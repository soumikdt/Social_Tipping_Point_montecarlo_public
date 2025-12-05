"""
Structured JSON logger using Python's standard logging library.

This module provides industry-standard structured logging with JSON output,
following Python logging best practices.
"""

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON lines."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as a JSON line."""
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'data'):
            log_data.update(record.data)
        
        return json.dumps(log_data, default=str)


def setup_json_logger(
    name: str,
    log_file: Path,
    level: int = logging.INFO,
    max_bytes: int = 50 * 1024 * 1024,  # 50MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a structured JSON logger following Python logging best practices.
    
    Parameters
    ----------
    name : str
        Logger name (use hierarchical names like 'simulation.weights')
    log_file : Path
        Path to the log file
    level : int
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    max_bytes : int
        Maximum size of log file before rotation
    backup_count : int
        Number of backup files to keep
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Ensure directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get logger (creates hierarchy automatically)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Don't propagate to root logger
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create rotating file handler
    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    handler.setLevel(level)
    
    # Set JSON formatter
    formatter = JSONFormatter()
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger


def log_structured(logger: logging.Logger, level: int, data: Dict[str, Any]) -> None:
    """
    Log structured data as JSON.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    level : int
        Log level (logging.INFO, logging.DEBUG, etc.)
    data : Dict[str, Any]
        Structured data to log
    """
    # Create log record with extra data
    extra = {'data': data}
    logger.log(level, '', extra=extra)

