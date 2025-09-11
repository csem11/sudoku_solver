#!/usr/bin/env python3
"""
Sudoku Solver - Simple grid detection from camera
"""

import cv2 as cv
import numpy as np
from pathlib import Path
from src import Sudoku, VideoCapture, FrameProcessor, GridDetector, GridTransformer, CellExtractor


def main():
    """Main function - simple and straightforward."""
    
    print("Starting Sudoku Solver...")
    print("Press 'q' to quit, 's' to save frame")
    
    # Initialize components
    frame_processor = FrameProcessor()
    grid_saved = False  # Track if we've already saved a grid
    print("✓ Ready to detect grids!")
    
    # Start camera
    with VideoCapture() as video:
        while True:
            # Read frame from camera
            success, frame = video.read_frame()
            if not success:
                print("Failed to read frame")
                break
            
            # Process the frame
            grid_saved = process_frame(frame, frame_processor, grid_saved)
            
            # Check for key press
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_frame(frame)
    
    # Clean up
    cv.destroyAllWindows()
    print("Done!")


def process_frame(frame, frame_processor, grid_saved):
    """Process one frame to detect grids."""
    
    # Make a copy for drawing
    display_frame = frame.copy()
    
    # Convert to grayscale and find edges
    processed = frame_processor.preprocess(frame)
    
    grid_detector = GridDetector(processed)
    grid_contours = grid_detector.find_grid()
    
    # Draw results
    if grid_contours is not None and len(grid_contours) > 0:
        print("Grid found!")
        
        # Draw green outline around grid
        cv.drawContours(display_frame, [grid_contours], -1, (0, 255, 0), 3)
        
        # Draw blue box around grid
        x, y, w, h = cv.boundingRect(grid_contours)
        cv.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Show grid info
        area = cv.contourArea(grid_contours)
        cv.putText(display_frame, f"Grid Area: {area:.0f}", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Apply grid transformation and test cell extraction
        try:
            # Get the 4 corners from the contour
            perimeter = cv.arcLength(grid_contours, True)
            epsilon = 0.02 * perimeter
            approx = cv.approxPolyDP(grid_contours, epsilon, True)
            
            if len(approx) == 4:
                # Transform the grid to a square
                grid_transformer = GridTransformer()
                transformed_grid = grid_transformer.transform(frame, approx)
                
                # Test cell extraction
                cell_extractor = CellExtractor(transformed_grid)
                print("Testing cell extraction...")
                
                # Extract and print cell information
                cells = cell_extractor.extract_cells()
                if cells:
                    print(f"Extracted {len(cells)} cells")
                    for i, cell in enumerate(cells):
                        if cell is not None:
                            print(f"Cell {i}: Shape {cell.shape}, Type {cell.dtype}")
                        else:
                            print(f"Cell {i}: None")
                    
                    # Save all cells to files for debugging (only once)
                    if not grid_saved:
                        cell_extractor.save_cells(cells)
                        grid_saved = True
                        print("✓ Grid cells saved to cells/ directory")
                    
                else:
                    print("No cells extracted")
                
                # Show transformed grid
                cv.imshow("Transformed Grid", transformed_grid)
                
        except Exception as e:
            print(f"Grid transformation failed: {e}")
        
    else:
        # No grid found
        cv.putText(display_frame, "No Grid", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Show both images side by side
    show_results(display_frame, processed)
    
    return grid_saved


def show_results(original, processed):
    """Show original and processed images side by side."""
    
    if len(processed.shape) == 2:
        processed_color = cv.cvtColor(processed, cv.COLOR_GRAY2BGR)
    else:
        processed_color = processed
    

    combined = np.hstack([original, processed_color])
    

    cv.imshow("Sudoku Solver - Original | Processed", combined)


def save_frame(frame):
    """Save current frame to file."""
    filename = f"frame_{len(list(Path('.').glob('frame_*.jpg')))}.jpg"
    cv.imwrite(filename, frame)
    print(f"Saved: {filename}")


if __name__ == "__main__":
    main()




