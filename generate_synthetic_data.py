#!/usr/bin/env python3
"""
Command-line interface for synthetic data generation
Usage: python generate_synthetic_data.py [--n-samples N] [--mode append|overwrite]
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.scripts.synethic_data_generator import SyntheticDataGenerator


def main():
    import argparse
    
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
    
    print("=== Synthetic Data Generator ===")
    print(f"Manual data directory: {args.manual_dir}")
    print(f"Synthetic data directory: {args.synthetic_dir}")
    print(f"Number of samples per digit: {args.n_samples_per_digit}")
    print(f"Mode: {args.mode}")
    print()
    
    # Create generator
    generator = SyntheticDataGenerator(args.manual_dir, args.synthetic_dir)
    
    if not generator.manual_images:
        print("No manual images found!")
        print("Please run manual digit collection first:")
        print("  python run_manual_digit_collection.py")
        return
    
    print(f"Found {len(generator.manual_images)} manual digit images")
    
    # Generate synthetic data
    generator.generate_synthetic_data(
        n_samples_per_digit=args.n_samples_per_digit,
        mode=args.mode,
        overwrite=(args.mode == "overwrite")
    )
    
    print("\nSynthetic data generation complete!")


if __name__ == "__main__":
    main()
