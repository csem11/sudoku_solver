#!/usr/bin/env python3
"""
Sudoku Grid Capture Tool

This tool allows users to capture a Sudoku grid from video feed and save
unprocessed cell images for training or analysis purposes.
"""

import cv2 as cv
import numpy as np
import os
import time
from datetime import datetime
from typing import Optional, List, Tuple

from src.capture.video_capture import VideoCapture
from src.capture.frame_processor import FrameProcessor
from src.detection.grid_detector import GridDetector
from src.detection.grid_transformer import GridTransformer
from src.detection.cell_extactor import CellExtractor


class GridCaptureTool:
    def __init__(self, output_dir: str = "data/digits/raw_cells", cell_size: int = 50):
        """
        Initialize the grid capture tool
        
        Args:
            output_dir: Directory to save captured cell images
            cell_size: Size of each cell in the transformed grid
        """
        self.output_dir = output_dir
        self.cell_size = cell_size
        
        # Initialize components
        self.video_capture = VideoCapture()
        self.frame_processor = FrameProcessor()
        self.grid_detector = GridDetector()
        self.grid_transformer = GridTransformer()  # Use default output_size
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # State tracking
        self.captured_grids = 0
        self.current_grid = None
        self.current_cells = None
        
    def start_capture_session(self) -> bool:
        """Start the video capture session"""
        return self.video_capture.start()
    
    def stop_capture_session(self):
        """Stop the video capture session"""
        self.video_capture.stop()
    
    def detect_and_transform_grid(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect grid in frame and transform to square
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Transformed grid if detected, None otherwise
        """
        # Preprocess frame
        processed_frame = self.frame_processor.preprocess(frame)
        
        # Detect grid
        grid_contour = self.grid_detector.detect(processed_frame)
        if grid_contour is None:
            return None
        
        # Transform grid to square
        transformed_grid = self.grid_transformer.transform(frame, grid_contour)
        
        return transformed_grid
    
    def extract_cells(self, grid: np.ndarray) -> List[np.ndarray]:
        """
        Extract individual cells from the transformed grid
        
        Args:
            grid: Transformed square grid image
            
        Returns:
            List of cell images (81 total for 9x9 grid)
        """
        # Calculate the expected grid size based on cell_size
        expected_size = 9 * self.cell_size
        
        # Resize grid to match expected size for cell extraction
        resized_grid = cv.resize(grid, (expected_size, expected_size))
        
        # Extract cells using the src module
        cell_extractor = CellExtractor(resized_grid)
        # Use the new attribute structure - get processed cells
        cells = cell_extractor.processed_cells
        
        return cells
    
    def save_cells(self, cells: List[np.ndarray], session_name: str = None) -> str:
        """
        Save extracted cells to organized directory structure
        
        Args:
            cells: List of cell images
            session_name: Optional name for this capture session
            
        Returns:
            Path to the saved session directory
        """
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"grid_capture_{timestamp}"
        
        session_dir = os.path.join(self.output_dir, session_name)
        os.makedirs(session_dir, exist_ok=True)
        
        saved_count = 0
        for i, cell in enumerate(cells):
            if cell is not None:
                row = i // 9
                col = i % 9
                filename = f"cell_{row:02d}_{col:02d}.png"
                filepath = os.path.join(session_dir, filename)
                
                # Save as unprocessed image (original cell)
                cv.imwrite(filepath, cell)
                saved_count += 1
        
        print(f"Saved {saved_count} cells to {session_dir}")
        return session_dir
    
    def run_interactive_capture(self):
        """Run interactive capture session with live preview"""
        if not self.start_capture_session():
            print("Failed to start video capture")
            return
        
        print("Grid Capture Tool - Interactive Mode")
        print("Commands:")
        print("  'c' - Capture current grid")
        print("  's' - Save current cells")
        print("  'q' - Quit")
        print("  'h' - Show this help")
        
        try:
            while True:
                success, frame = self.video_capture.read_frame()
                if not success:
                    print("Failed to read frame")
                    break
                
                # Detect and transform grid
                transformed_grid = self.detect_and_transform_grid(frame)
                
                # Create display frame
                display_frame = frame.copy()
                
                if transformed_grid is not None:
                    # Show transformed grid in corner
                    small_grid = cv.resize(transformed_grid, (200, 200))
                    h, w = display_frame.shape[:2]
                    display_frame[h-220:h-20, w-220:w-20] = small_grid
                    
                    # Draw border around preview
                    cv.rectangle(display_frame, (w-220, h-220), (w-20, h-20), (0, 255, 0), 2)
                    
                    # Store current grid for saving
                    self.current_grid = transformed_grid
                
                # Show frame
                cv.imshow('Sudoku Grid Capture', display_frame)
                
                # Handle key presses
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    if self.current_grid is not None:
                        print("Capturing grid...")
                        self.current_cells = self.extract_cells(self.current_grid)
                        print(f"Extracted {len([c for c in self.current_cells if c is not None])} cells")
                    else:
                        print("No grid detected. Position the Sudoku grid in view.")
                elif key == ord('s'):
                    if self.current_cells is not None:
                        session_dir = self.save_cells(self.current_cells)
                        self.captured_grids += 1
                        print(f"Grid saved! Total captures: {self.captured_grids}")
                    else:
                        print("No cells to save. Capture a grid first with 'c'.")
                elif key == ord('h'):
                    print("\nCommands:")
                    print("  'c' - Capture current grid")
                    print("  's' - Save current cells")
                    print("  'q' - Quit")
                    print("  'h' - Show this help")
        
        finally:
            self.stop_capture_session()
            cv.destroyAllWindows()
    
    def capture_single_grid(self, timeout: float = 10.0) -> Optional[str]:
        """
        Capture a single grid automatically
        
        Args:
            timeout: Maximum time to wait for grid detection (seconds)
            
        Returns:
            Path to saved session directory if successful, None otherwise
        """
        if not self.start_capture_session():
            print("Failed to start video capture")
            return None
        
        print(f"Looking for Sudoku grid (timeout: {timeout}s)...")
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                success, frame = self.video_capture.read_frame()
                if not success:
                    continue
                
                # Detect and transform grid
                transformed_grid = self.detect_and_transform_grid(frame)
                
                if transformed_grid is not None:
                    print("Grid detected! Extracting cells...")
                    cells = self.extract_cells(transformed_grid)
                    session_dir = self.save_cells(cells)
                    print(f"Grid captured and saved to: {session_dir}")
                    return session_dir
                
                # Show preview
                cv.imshow('Looking for Grid...', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.stop_capture_session()
            cv.destroyAllWindows()
        
        print("No grid detected within timeout period")
        return None


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sudoku Grid Capture Tool")
    parser.add_argument("--output-dir", default="data/digits/raw_cells", 
                       help="Directory to save captured cells")
    parser.add_argument("--cell-size", type=int, default=50,
                       help="Size of each cell in pixels")
    parser.add_argument("--mode", choices=["interactive", "single"], default="interactive",
                       help="Capture mode: interactive or single")
    parser.add_argument("--timeout", type=float, default=10.0,
                       help="Timeout for single capture mode (seconds)")
    
    args = parser.parse_args()
    
    # Create capture tool
    tool = GridCaptureTool(output_dir=args.output_dir, cell_size=args.cell_size)
    
    if args.mode == "interactive":
        tool.run_interactive_capture()
    else:
        tool.capture_single_grid(timeout=args.timeout)


if __name__ == "__main__":
    main()
