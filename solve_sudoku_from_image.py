#!/usr/bin/env python3
"""
Main script to solve Sudoku puzzles from images using the integrated solver.
"""

import cv2 as cv
import numpy as np
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.generation.integrated_solver import IntegratedSudokuSolver

def main():
    """Main function to demonstrate the integrated Sudoku solver."""
    
    # Initialize the solver
    print("Initializing Sudoku solver...")
    solver = IntegratedSudokuSolver()
    
    # Check if image path is provided as command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use a default image from the data directory
        image_path = "data/grids/grid_0.jpg"
        if not Path(image_path).exists():
            print(f"Default image {image_path} not found.")
            print("Usage: python solve_sudoku_from_image.py <image_path>")
            print("Or place a Sudoku image in data/grids/ and update the default path.")
            return
    
    if not Path(image_path).exists():
        print(f"Image file {image_path} not found.")
        return
    
    print(f"Processing image: {image_path}")
    
    try:
        # Solve the Sudoku puzzle
        original_image, original_puzzle, solved_puzzle = solver.solve_from_image(
            image_path, 
            show_comparison=True, 
            save_result=True,
            output_path=f"solved_{Path(image_path).name}"
        )
        
        print("\n" + "="*50)
        print("SOLUTION COMPLETE")
        print("="*50)
        
        # Display final statistics
        stats = solver.grid_generator.get_prediction_stats()
        print(f"\nProcessing Statistics:")
        print(f"- Total cells processed: 81")
        print(f"- Blank cells detected: {stats['blank_cells_detected']}")
        print(f"- Neural network predictions: {stats['neural_net_predictions']}")
        print(f"- Average confidence: {stats['average_confidence']:.3f}")
        print(f"- High confidence predictions: {stats['high_confidence_count']}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

def process_camera():
    """Process Sudoku puzzles from camera feed."""
    print("Initializing camera solver...")
    solver = IntegratedSudokuSolver()
    
    print("Starting camera feed...")
    print("Instructions:")
    print("- Position a Sudoku puzzle in front of the camera")
    print("- Press 's' to solve the current frame")
    print("- Press 'q' to quit")
    
    solver.process_video_stream()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--camera":
        process_camera()
    else:
        main()
