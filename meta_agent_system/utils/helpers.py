import os
import json
from typing import Dict, Any, List, Optional
import time

def ensure_directory_exists(directory_path: str):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def save_json(data: Dict[str, Any], filepath: str):
    """Save data as JSON to a file"""
    # Ensure directory exists
    directory = os.path.dirname(filepath)
    ensure_directory_exists(directory)
    
    # Save data
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON data from a file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def format_time(seconds: float) -> str:
    """Format time in seconds to a readable string"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def get_timestamp() -> str:
    """Get current timestamp string in format suitable for filenames"""
    return time.strftime("%Y%m%d_%H%M%S")

def truncate_string(text: str, max_length: int = 100) -> str:
    """Truncate string to max_length and add ellipsis if needed"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."
