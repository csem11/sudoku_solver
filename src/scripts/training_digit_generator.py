#!/usr/bin/env python3
"""
Training Digit Generator (Simple)
Uses CellExtractor.preprocess_cell to process and save digit images.
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.naming import parse_manual_filename, parse_synthetic_filename
from src.detection.cell_extactor import CellExtractor

def process_and_save_images(input_dir, output_dir, parse_filename_func, source, start_idx=0):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.jpg"))
    idx = start_idx
    
    # Create a dummy grid for CellExtractor initialization
    # We'll create a minimal 9x9 grid and place each digit in the center
    grid_size = 450  # 9 * 50 (cell_size)
    dummy_grid = np.full((grid_size, grid_size, 3), 255, dtype=np.uint8)
    cell_extractor = CellExtractor(dummy_grid)
    
    for file_path in files:
        metadata = parse_filename_func(file_path.name)
        if metadata is None:
            continue
        img = cv.imread(str(file_path))
        if img is None:
            continue
        
        # Convert to BGR if grayscale
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        
        # Place the digit in the center of the dummy grid
        center_start = grid_size // 2 - 25  # Center the 50x50 cell
        dummy_grid[center_start:center_start+50, center_start:center_start+50] = 255  # Clear
        # Resize and place the digit
        resized_img = cv.resize(img, (50, 50))
        dummy_grid[center_start:center_start+50, center_start:center_start+50] = resized_img
        
        # Extract the center cell (index 40 = row 4, col 4)
        center_cell = cell_extractor.cells[40]
        if center_cell is not None:
            processed = cell_extractor.preprocess_cell(center_cell)
        else:
            # Fallback: just resize the original image
            processed = cv.resize(img, (28, 28))
        
        outname = f"train_{metadata['digit']}_g{metadata['grid_id']}_c{metadata['cell_id']}_{source}_{idx:03d}.jpg"
        cv.imwrite(str(output_dir / outname), processed)
        idx += 1
    return idx

def main():
    parser = argparse.ArgumentParser(description="Generate training data from manual and synthetic digits (simple)")
    parser.add_argument("--mode", choices=["append", "overwrite"], default="append")
    parser.add_argument("--manual-dir", type=str, default="data/digits/manual")
    parser.add_argument("--synthetic-dir", type=str, default="data/digits/synthetic")
    parser.add_argument("--training-dir", type=str, default="data/digits/training")
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    if args.mode == "overwrite":
        for f in training_dir.glob("*.jpg"):
            f.unlink()
        start_idx = 0
    else:
        existing = list(training_dir.glob("train_*.jpg"))
        if existing:
            indices = []
            for f in existing:
                parts = f.stem.split("_")
                if len(parts) >= 6:
                    try:
                        indices.append(int(parts[-1]))
                    except Exception:
                        pass
            start_idx = max(indices, default=-1) + 1
        else:
            start_idx = 0

    idx = start_idx
    if Path(args.manual_dir).exists():
        idx = process_and_save_images(args.manual_dir, args.training_dir, parse_manual_filename, "manual", idx)
    if Path(args.synthetic_dir).exists():
        process_and_save_images(args.synthetic_dir, args.training_dir, parse_synthetic_filename, "synthetic", idx)

if __name__ == "__main__":
    main()