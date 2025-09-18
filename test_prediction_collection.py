#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced manual digit collection with AI predictions
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.scripts.manual_digit_collection import ManualDigitCollector

def main():
    print("=== Testing Enhanced Manual Digit Collection ===")
    print("This tool now includes AI predictions to speed up the labeling process!")
    print()
    print("New Features:")
    print("- AI predictions are automatically generated after grid capture")
    print("- Pre-populated digits can be reviewed and corrected")
    print("- Confidence levels are displayed to help identify likely errors")
    print("- Color-coded grid overview shows prediction confidence")
    print("- Statistics show prediction accuracy and corrections made")
    print("- Automatic window positioning to prevent overlap")
    print()
    print("Controls:")
    print("- Press 'p' during capture to toggle AI predictions on/off")
    print("- Press 'r' during labeling to reset a cell to AI prediction")
    print("- Press 'i' during labeling to show prediction statistics")
    print("- Use A/D/W/S to navigate between cells")
    print("- Press number keys (0-9) to assign digits")
    print("- Press 'F' to finish and save, 'Q' to quit")
    print()
    
    try:
        collector = ManualDigitCollector()
        collector.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
