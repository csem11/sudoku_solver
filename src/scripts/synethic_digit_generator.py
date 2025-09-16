#!/usr/bin/env python3
"""
Synthetic Data Generator for Sudoku Digits
Creates augmented versions of manually collected digit images through perspective and shading distortions
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import argparse
import random
import os
import sys
from typing import List, Tuple, Optional

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.naming import parse_manual_filename, generate_synthetic_filename



class SyntheticDataGenerator:
    def __init__(self, manual_data_dir: str = "data/digits/manual", 
                 synthetic_data_dir: str = "data/digits/synthetic"):
        self.manual_data_dir = Path(manual_data_dir)
        self.synthetic_data_dir = Path(synthetic_data_dir)
        self.synthetic_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing manual data
        self.manual_images = self._load_manual_images()
        print(f"Loaded {len(self.manual_images)} manual digit images")
        
    def _load_manual_images(self) -> List[Tuple[np.ndarray, int, str, int, int]]:
        """Load all manual digit images with their labels and metadata"""
        images = []
        
        if not self.manual_data_dir.exists():
            print(f"Manual data directory {self.manual_data_dir} does not exist!")
            return images
            
        for file_path in self.manual_data_dir.glob("*.jpg"):
            # Parse filename using new naming convention
            metadata = parse_manual_filename(file_path.name)
            if metadata is None:
                continue
                
            # Load image
            img = cv.imread(str(file_path), cv.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((
                    img, 
                    metadata['digit'], 
                    file_path.name,
                    metadata['grid_id'],
                    metadata['cell_id']
                ))
                
        return images
    
    def _perspective_transform(self, img: np.ndarray, max_shift: float = 0.03) -> np.ndarray:
        """Apply random perspective transformation to simulate camera angle variations"""
        h, w = img.shape[:2]
        
        # Define original corners
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Add random shifts to corners (reduced from 0.1 to 0.03)
        shift_range = int(min(w, h) * max_shift)
        new_corners = corners.copy()
        
        # Randomly shift each corner
        for i in range(4):
            dx = random.randint(-shift_range, shift_range)
            dy = random.randint(-shift_range, shift_range)
            new_corners[i][0] = max(0, min(w, new_corners[i][0] + dx))
            new_corners[i][1] = max(0, min(h, new_corners[i][1] + dy))
        
        # Calculate perspective transform matrix
        matrix = cv.getPerspectiveTransform(corners, new_corners)
        
        # Apply transformation
        transformed = cv.warpPerspective(img, matrix, (w, h))
        
        return transformed
    
    def _shading_variations(self, img: np.ndarray) -> np.ndarray:
        """Apply various shading and lighting variations"""
        result = img.copy()
        
        # Random brightness adjustment (reduced range)
        brightness = random.uniform(-15, 15)
        result = cv.add(result, brightness)
        
        # Random contrast adjustment (reduced range)
        contrast = random.uniform(0.85, 1.15)
        result = cv.multiply(result, contrast)
        
        # Random gamma correction (reduced range)
        gamma = random.uniform(0.9, 1.1)
        result = np.power(result / 255.0, gamma) * 255.0
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Random noise (reduced probability and intensity)
        if random.random() < 0.15:  # 15% chance of adding noise (reduced from 30%)
            noise = np.random.normal(0, random.uniform(2, 8), result.shape)
            result = cv.add(result, noise.astype(np.uint8))
        
        # Random blur (reduced probability and only slight blur)
        if random.random() < 0.1:  # 10% chance of adding blur (reduced from 20%)
            kernel_size = 3  # Only use 3x3 kernel
            result = cv.GaussianBlur(result, (kernel_size, kernel_size), 0)
        
        return result
    
    def _rotation_transform(self, img: np.ndarray, max_angle: float = 3.0) -> np.ndarray:
        """Apply slight rotation to simulate hand positioning variations"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Random rotation angle (reduced from 10.0 to 3.0 degrees)
        angle = random.uniform(-max_angle, max_angle)
        
        # Get rotation matrix
        matrix = cv.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv.warpAffine(img, matrix, (w, h), borderMode=cv.BORDER_REFLECT)
        
        return rotated
    
    def _scale_transform(self, img: np.ndarray, scale_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
        """Apply slight scaling variations"""
        h, w = img.shape[:2]
        
        # Random scale factor (reduced range from 0.9-1.1 to 0.95-1.05)
        scale = random.uniform(scale_range[0], scale_range[1])
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        scaled = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_CUBIC)
        
        # Pad or crop to original size
        if scale > 1.0:
            # Crop from center
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            result = scaled[start_y:start_y+h, start_x:start_x+w]
        else:
            # Pad to original size
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            result = cv.copyMakeBorder(scaled, pad_y, h-new_h-pad_y, 
                                     pad_x, w-new_w-pad_x, cv.BORDER_CONSTANT, value=255)
        
        return result
    
    def generate_synthetic_image(self, img: np.ndarray) -> np.ndarray:
        """Generate a synthetic version of the input image with multiple transformations"""
        result = img.copy()
        
        # Apply transformations in random order
        transforms = [
            self._perspective_transform,
            self._shading_variations,
            self._rotation_transform,
            self._scale_transform
        ]
        
        # Randomly select 1-3 transformations to apply (reduced from 2-4)
        num_transforms = random.randint(1, 3)
        selected_transforms = random.sample(transforms, num_transforms)
        
        for transform in selected_transforms:
            result = transform(result)
        
        return result
    
    def generate_synthetic_data(self, n_samples_per_digit: int = 100, mode: str = "append", 
                              overwrite: bool = False) -> None:
        """Generate synthetic digit images with balanced distribution"""
        if not self.manual_images:
            print("No manual images found! Please collect manual data first.")
            return
        
        # Group manual images by digit
        digit_groups = {}
        for img, digit, filename, grid_id, cell_id in self.manual_images:
            if digit not in digit_groups:
                digit_groups[digit] = []
            digit_groups[digit].append((img, digit, filename, grid_id, cell_id))
        
        print(f"Found manual images for digits: {sorted(digit_groups.keys())}")
        for digit in sorted(digit_groups.keys()):
            print(f"  Digit {digit}: {len(digit_groups[digit])} images")
        
        # Determine output directory and file naming
        if mode == "overwrite" or overwrite:
            # Clear existing synthetic data
            for file_path in self.synthetic_data_dir.glob("*.jpg"):
                file_path.unlink()
            print("Cleared existing synthetic data")
            start_idx = 0
        else:
            # Find next available index for appending
            existing_files = list(self.synthetic_data_dir.glob("synthetic_*.jpg"))
            if existing_files:
                indices = []
                for file_path in existing_files:
                    try:
                        idx = int(file_path.stem.split("_")[1])
                        indices.append(idx)
                    except (ValueError, IndexError):
                        continue
                start_idx = max(indices, default=-1) + 1
            else:
                start_idx = 0
        
        total_samples = n_samples_per_digit * len(digit_groups)
        print(f"Generating {n_samples_per_digit} samples per digit ({total_samples} total)...")
        print(f"Starting from index {start_idx}")
        
        # Generate synthetic images for each digit
        generated_count = 0
        current_idx = start_idx
        
        for digit in sorted(digit_groups.keys()):
            digit_images = digit_groups[digit]
            print(f"Generating {n_samples_per_digit} samples for digit {digit}...")
            
            for i in range(n_samples_per_digit):
                # Randomly select a manual image for this digit
                original_img, _, original_filename, grid_id, cell_id = random.choice(digit_images)
                
                # Generate synthetic version
                synthetic_img = self.generate_synthetic_image(original_img)
                
                # Save synthetic image with new naming convention
                synthetic_filename = generate_synthetic_filename(digit, grid_id, cell_id, i)
                output_path = self.synthetic_data_dir / synthetic_filename
                cv.imwrite(str(output_path), synthetic_img)
                
                generated_count += 1
                current_idx += 1
                
                if (i + 1) % 20 == 0:
                    print(f"  Generated {i + 1}/{n_samples_per_digit} for digit {digit}")
        
        print(f"Successfully generated {generated_count} synthetic images!")
        print(f"Synthetic data saved to: {self.synthetic_data_dir}")
        
        # Generate summary
        self._generate_summary()
    
    def _generate_summary(self) -> None:
        """Generate a summary of the synthetic data"""
        synthetic_files = list(self.synthetic_data_dir.glob("syn_*.jpg"))
        
        # Count digits using new naming convention
        digit_counts = {}
        for file_path in synthetic_files:
            metadata = parse_manual_filename(file_path.name)  # This will work for both manual and synthetic
            if metadata is None:
                # Try parsing as synthetic
                from src.utils.naming import parse_synthetic_filename
                metadata = parse_synthetic_filename(file_path.name)
            
            if metadata is not None:
                digit = metadata['digit']
                digit_counts[digit] = digit_counts.get(digit, 0) + 1
        
        # Write summary
        summary_path = self.synthetic_data_dir / "synthetic_data_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Synthetic Data Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Total synthetic images: {len(synthetic_files)}\n")
            f.write(f"Generated from manual data: {len(self.manual_images)} images\n\n")
            f.write("Digit distribution:\n")
            for digit in sorted(digit_counts.keys()):
                f.write(f"Digit {digit}: {digit_counts[digit]} images\n")
        
        print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic digit data from manual collection")
    parser.add_argument("--n-samples-per-digit", type=int, default=100, 
                       help="Number of synthetic samples to generate per digit (default: 100)")
    parser.add_argument("--mode", choices=["append", "overwrite"], default="append",
                       help="Mode: append to existing data or overwrite (default: append)")
    parser.add_argument("--manual-dir", type=str, default="data/digits/manual",
                       help="Directory containing manual digit data (default: data/digits/manual)")
    parser.add_argument("--synthetic-dir", type=str, default="data/digits/synthetic",
                       help="Directory to save synthetic data (default: data/digits/synthetic)")
    
    args = parser.parse_args()
    
    # Create generator
    generator = SyntheticDataGenerator(args.manual_dir, args.synthetic_dir)
    
    # Generate synthetic data
    generator.generate_synthetic_data(
        n_samples_per_digit=args.n_samples_per_digit,
        mode=args.mode,
        overwrite=(args.mode == "overwrite")
    )


if __name__ == "__main__":
    main()
