"""
Script modules for data collection and generation
"""

from .manual_digit_collection import collect_manual_digits
from .synethic_digit_generator import generate_synthetic_digits

__all__ = [
    'collect_manual_digits',
    'generate_synthetic_digits'
]
