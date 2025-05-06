import logging
import colorlog
import os
from datetime import datetime
import sys
import re

# Define a filter to suppress specific warning messages
class SuppressSpecificWarningsFilter(logging.Filter):
    def filter(self, record):
        # Allow all non-warning messages through
        if record.levelno != logging.WARNING:
            return True
            
        # Check if this is a rule format warning we want to suppress
        if record.levelno == logging.WARNING and hasattr(record, 'msg'):
            if "Unrecognized rule format" in record.msg or "Unrecognized condition" in record.msg:
                return False  # Suppress these specific warnings
                
        return True  # Allow all other warnings

# Store loggers in a dictionary to avoid creating duplicates
_loggers = {}

def setup_logger(name, log_level=logging.INFO, log_file=None):
    """
    Set up a logger with color formatting for console and optionally file logging.
    
    Args:
        name: Name of the logger
        log_level: Logging level (default: INFO)
        log_file: Optional file path to save logs
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers if any
    if logger.handlers:
        # Properly close handlers before clearing
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logger.handlers.clear()
    
    # Create color formatter for console
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create file formatter
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name):
    """
    Get a logger with the specified name.
    
    Args:
        name: The name of the logger
        
    Returns:
        A logger object
    """
    if name in _loggers:
        return _loggers[name]
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Stop propagation to avoid duplicate logs
    logger.propagate = False
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create handlers
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Add the filter to suppress specific warnings
    if name == "meta_agent_system.experts.validator":
        console_handler.addFilter(SuppressSpecificWarningsFilter())
    
    # Add formatters
    console_format = '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    console_formatter = colorlog.ColoredFormatter(
        console_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    
    # Save logger for reuse
    _loggers[name] = logger
    
    return logger

def get_logger_for_module(module_name):
    """Get a logger for a specific module"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = f"{log_dir}/meta_agent_{date_str}.log"
    
    return setup_logger(module_name, log_file=log_file)
