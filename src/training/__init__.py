"""
Training modules for the Sudoku solver project
"""

from .dataset import retrieve_digit_dataset
from .model import DigitClassifier

__all__ = [
    'retrieve_digit_dataset',
    'DigitClassifier'
]