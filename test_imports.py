#!/usr/bin/env python3
"""
Simple test to make sure all imports work
"""
import numpy as np

def test_imports():
    """Test that we can import all the modules."""
    print("Testing imports...")
    
    try:
        # Test main package import
        from src import Sudoku, VideoCapture, FrameProcessor, GridDetector
        print("✓ All modules imported successfully")
        
        # Test creating instances
        from src.capture.frame_processor import FrameProcessor as FP
        from src.detection.grid_detector import GridDetector as GD
        
        frame_processor = FP()
        grid_detector = GD(frame_processor.preprocess(np.zeros((100, 100, 3), dtype=np.uint8)))
        print("✓ Objects created successfully")
        
        print("\n🎉 Everything is working!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\nYou can now run: python main.py")
    else:
        print("\nFix the import issues first.")
