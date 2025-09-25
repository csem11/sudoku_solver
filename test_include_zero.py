#!/usr/bin/env python3
"""
Test script to demonstrate the include_zero functionality.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from training.dataset import retrieve_digit_dataset

def test_dataset_with_without_zero():
    """Test dataset loading with and without zero digit."""
    
    # Get data directory
    data_dir = Path(__file__).parent / "data" / "digits"
    
    print("=== Testing Dataset with include_zero=False (default) ===")
    X_no_zero, y_no_zero = retrieve_digit_dataset(
        str(data_dir), 
        return_categorical=True, 
        include_zero=False
    )
    
    if len(X_no_zero) > 0:
        print(f"Loaded {len(X_no_zero)} samples without zero digit")
        print(f"Label shape: {y_no_zero.shape}")
        print(f"Number of classes: {y_no_zero.shape[1]}")
    else:
        print("No data loaded")
    
    print("\n=== Testing Dataset with include_zero=True ===")
    X_with_zero, y_with_zero = retrieve_digit_dataset(
        str(data_dir), 
        return_categorical=True, 
        include_zero=True
    )
    
    if len(X_with_zero) > 0:
        print(f"Loaded {len(X_with_zero)} samples with zero digit")
        print(f"Label shape: {y_with_zero.shape}")
        print(f"Number of classes: {y_with_zero.shape[1]}")
    else:
        print("No data loaded")

if __name__ == "__main__":
    test_dataset_with_without_zero()
