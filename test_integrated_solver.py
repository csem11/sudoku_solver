#!/usr/bin/env python3
"""
Test script for the integrated Sudoku solver.
"""

import cv2 as cv
import numpy as np
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.generation.integrated_solver import IntegratedSudokuSolver
from src.generation.sudoku import Sudoku

def test_sudoku_solver():
    """Test the Sudoku solver with a known puzzle."""
    print("Testing Sudoku solver with a known puzzle...")
    
    # Create a test puzzle (easy one from the sudoku.py file)
    easy_board = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])
    
    # Test the Sudoku class directly
    sudoku = Sudoku(board=easy_board.copy())
    print("Original puzzle:")
    sudoku.print_board()
    
    print("\nSolving puzzle...")
    sudoku.solve_board()
    
    print("\nSolved puzzle:")
    sudoku.print_board()

def test_visualizer():
    """Test the board visualizer."""
    print("\nTesting board visualizer...")
    
    from src.generation.board_visualizer import SudokuBoardVisualizer
    
    # Create test puzzles
    original_puzzle = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])
    
    solved_puzzle = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])
    
    visualizer = SudokuBoardVisualizer()
    visualizer.visualize_solution_comparison(original_puzzle, solved_puzzle, "Test Sudoku Solution")
    print("Visualization test completed.")

def test_integrated_solver_with_sample_image():
    """Test the integrated solver with a sample image if available."""
    print("\nTesting integrated solver...")
    
    # Check if we have any sample images
    sample_images = list(Path("data/grids").glob("*.jpg"))
    
    if not sample_images:
        print("No sample images found in data/grids/ directory.")
        print("Please add some Sudoku images to test the integrated solver.")
        return
    
    # Use the first available image
    image_path = str(sample_images[0])
    print(f"Testing with image: {image_path}")
    
    try:
        solver = IntegratedSudokuSolver()
        original_image, original_puzzle, solved_puzzle = solver.solve_from_image(
            image_path, 
            show_comparison=True, 
            save_result=True
        )
        
        if original_puzzle is not None and solved_puzzle is not None:
            print("Integrated solver test completed successfully!")
        elif original_puzzle is not None:
            print("Integrated solver detected puzzle but could not solve it.")
        else:
            print("Integrated solver test failed - no puzzle detected in image.")
            print("This may be normal if the image doesn't contain a clear Sudoku grid.")
            
    except Exception as e:
        print(f"Error testing integrated solver: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests."""
    print("="*60)
    print("SUDOKU SOLVER INTEGRATION TESTS")
    print("="*60)
    
    # Test 1: Basic Sudoku solver
    test_sudoku_solver()
    
    # Test 2: Board visualizer
    test_visualizer()
    
    # Test 3: Integrated solver (if sample images available)
    test_integrated_solver_with_sample_image()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
