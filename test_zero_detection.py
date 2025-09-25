#!/usr/bin/env python3
"""
Test script to verify the optimized zero detection implementation
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append('src')

from detection.cell_extactor import CellExtractor

def test_zero_detection():
    """Test the optimized zero detection on sample images"""
    
    print("=== Testing Optimized Zero Detection ===")
    
    # Load some sample processed images
    processed_dir = Path("data/digits/manual/processed")
    if not processed_dir.exists():
        print("Error: Processed directory not found!")
        return
    
    # Get some zero and non-zero images
    zero_images = list(processed_dir.glob("0_g*_c*_man_processed.jpg"))[:10]
    non_zero_images = []
    for digit in range(1, 10):
        non_zero_images.extend(list(processed_dir.glob(f"{digit}_g*_c*_man_processed.jpg"))[:2])
    
    print(f"Testing with {len(zero_images)} zero images and {len(non_zero_images)} non-zero images")
    
    # Create cell extractor
    extractor = CellExtractor(np.zeros((100, 100)))
    
    # Test zero detection
    correct_zeros = 0
    correct_non_zeros = 0
    total_zeros = len(zero_images)
    total_non_zeros = len(non_zero_images)
    
    print("\nTesting Zero Images:")
    for i, img_path in enumerate(zero_images):
        image = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        if image is None:
            continue
            
        is_blank = extractor.is_cell_blank(image)
        features = extractor.analyze_cell_features(image)
        
        if is_blank:
            correct_zeros += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"  {status} {img_path.name}: edges={features['edge_pixels']}, contours={features['num_contours']}, score={features['combined_score']:.2f}")
    
    print("\nTesting Non-Zero Images:")
    for i, img_path in enumerate(non_zero_images):
        image = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        if image is None:
            continue
            
        is_blank = extractor.is_cell_blank(image)
        features = extractor.analyze_cell_features(image)
        
        if not is_blank:
            correct_non_zeros += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"  {status} {img_path.name}: edges={features['edge_pixels']}, contours={features['num_contours']}, score={features['combined_score']:.2f}")
    
    # Calculate accuracy
    zero_accuracy = (correct_zeros / total_zeros) * 100 if total_zeros > 0 else 0
    non_zero_accuracy = (correct_non_zeros / total_non_zeros) * 100 if total_non_zeros > 0 else 0
    overall_accuracy = ((correct_zeros + correct_non_zeros) / (total_zeros + total_non_zeros)) * 100
    
    print(f"\n=== Results ===")
    print(f"Zero detection accuracy: {correct_zeros}/{total_zeros} ({zero_accuracy:.1f}%)")
    print(f"Non-zero detection accuracy: {correct_non_zeros}/{total_non_zeros} ({non_zero_accuracy:.1f}%)")
    print(f"Overall accuracy: {(correct_zeros + correct_non_zeros)}/{(total_zeros + total_non_zeros)} ({overall_accuracy:.1f}%)")
    
    # Performance test
    print(f"\n=== Performance Test ===")
    test_image = cv.imread(str(zero_images[0]), cv.IMREAD_GRAYSCALE)
    
    # Time the optimized method
    start_time = time.time()
    for _ in range(100):
        extractor.is_cell_blank(test_image)
    optimized_time = (time.time() - start_time) / 100
    
    print(f"Optimized method: {optimized_time*1000:.2f}ms per cell")
    print(f"Expected ~50% improvement over adaptive thresholding")
    
    return overall_accuracy > 90  # Return True if accuracy is good

if __name__ == "__main__":
    success = test_zero_detection()
    if success:
        print("\n✅ Zero detection optimization test PASSED!")
    else:
        print("\n❌ Zero detection optimization test FAILED!")
