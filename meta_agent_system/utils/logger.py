import logging
import colorlog
import os
from datetime import datetime

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

def get_logger(module_name):
    """Get a logger for a specific module"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    date_str = datetime.now().strftime("%Y-%m-%d")
    log_file = f"{log_dir}/meta_agent_{date_str}.log"
    
    return setup_logger(module_name, log_file=log_file)
