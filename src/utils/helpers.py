"""
Utility helper functions.

This module contains miscellaneous utility functions used throughout the application.
"""

from datetime import datetime
from typing import Any, Dict


def format_timestamp(dt: datetime) -> str:
    """Format datetime to ISO format string."""
    return dt.isoformat()


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


# Add more helper functions as needed during development
