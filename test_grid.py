#!/usr/bin/env python3
"""
Grid Testing Script - Interactive Sudoku grid detection and analysis
"""

from src import GridTester


def main():
    """Main function for grid testing."""
    print("Starting Sudoku Grid Tester...")
    
    # Initialize the grid tester
    tester = GridTester()
    
    # Run interactive testing
    tester.run_interactive_test()


if __name__ == "__main__":
    main()
