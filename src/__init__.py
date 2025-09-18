
"""
Sudoku Solver - Main package
"""

# Core modules
from .sudoku import Sudoku

# Detection modules
from .detection.grid_detector import GridDetector
from .detection.grid_transformer import GridTransformer
from .detection.cell_extactor import CellExtractor

# Capture modules
from .capture.video_capture import VideoCapture
from .capture.frame_processor import FrameProcessor

# Generation modules
from .generation.grid_generator import GridGenerator

# Training modules
from .training.model import DigitClassifier

# Testing modules
# from .testing.grid_tester import GridTester  # Commented out - module doesn't exist

# Utility modules
from .utils.naming import (
    parse_manual_filename,
    parse_synthetic_filename,
    parse_filename,
    generate_manual_filename,
    generate_synthetic_filename,
    get_cell_position,
    get_cell_id,
    validate_filename
)

__all__ = [
    # Core
    'Sudoku',
    
    # Detection
    'GridDetector',
    'GridTransformer', 
    'CellExtractor',
    
    # Capture
    'VideoCapture',
    'FrameProcessor',
    
    # Generation
    'GridGenerator',
    
    # Training
    'DigitClassifier',
    
    # Testing
    # 'GridTester',  # Commented out - module doesn't exist
    
    # Utils
    'parse_manual_filename',
    'parse_synthetic_filename',
    'parse_filename', 
    'generate_manual_filename',
    'generate_synthetic_filename',
    'get_cell_position',
    'get_cell_id',
    'validate_filename'
]