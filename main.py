#!/usr/bin/env python3
"""
Sudoku Solver - Simple grid detection from camera
"""

import cv2 as cv
import numpy as np
from pathlib import Path
from src import Sudoku, VideoCapture, FrameProcessor, GridDetector, GridTransformer, CellExtractor, GridGenerator


def main():
    print("Sudoku Detection and Solver")
    print("Controls:")
    print("  'c' - Capture current detected grid")
    print("  'r' - Reset captured grid")
    print("  'q' - Quit")
    print("Hold a Sudoku puzzle in front of the camera and press 'c' to capture it!")
    print()
    
    # Initialize video capture
    cap = VideoCapture(0)
    if not cap.start():
        print("Failed to start video capture.")
        return
    
    frame_processor = FrameProcessor()
    grid_detector = GridDetector()
    grid_transformer = GridTransformer(450)  # Use same size as manual tool for better predictions
    cell_extractor = None
    grid_generator = GridGenerator()

    detected_grid = None
    grid_img = None
    predicted_grid = None
    prob_grid = None
    captured_grid = None
    captured_prob_grid = None

    while True:
        ret, frame = cap.read_frame()
        if not ret:
            print("Failed to capture frame from camera.")
            break

        processed_frame = frame_processor.preprocess(frame)
        grid_contour = grid_detector.detect(processed_frame)

        display_frame = frame.copy()

        if grid_contour is not None:
            # Highlight the detected grid with corners and numbering
            display_frame = grid_detector.highlight_grid(display_frame, grid_contour)

            # Transform the grid to a top-down view
            grid_img = grid_transformer.transform(frame, grid_contour)

            # Show the warped grid image
            cv.imshow("Grid Image", grid_img)

            # Extract cells from the grid image (but don't predict yet)
            cell_extractor = CellExtractor(grid_img)
            cells = np.array(cell_extractor.processed_cells)
            # Ensure cells are in the right shape for the generator
            cells = np.array([cell for cell in cells if cell is not None])
            if len(cells) == 81:
                cells = cells.reshape(9, 9, 28, 28)
                cells = cells.reshape(81, 28, 28, 1)
                # Store cells for potential capture
                detected_grid = cells
        else:
            cv.destroyWindow("Grid Image")
            grid_img = None
            detected_grid = None

        # Add instructions to the frame
        instructions = [
            "Controls:",
            "'c' - Capture grid",
            "'r' - Reset",
            "'q' - Quit"
        ]
        
        y_offset = 30
        for instruction in instructions:
            cv.putText(display_frame, instruction, (10, y_offset), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.putText(display_frame, instruction, (10, y_offset), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_offset += 25

        # Show the frame with detected grid
        cv.imshow("Sudoku Detection", display_frame)

        # Show captured grid if available
        if captured_grid is not None:
            print("Captured Sudoku Grid:")
            print(captured_grid)
            # Show prediction statistics
            grid_generator.print_prediction_stats()
            # Visualize the board with probabilities and stats
            grid_generator.show_board(captured_grid, captured_prob_grid)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and detected_grid is not None:
            # Capture the current detected grid
            print("Capturing grid...")
            captured_grid, captured_prob_grid = grid_generator.generate_grid(detected_grid, include_prob_grid=True)
            print("Grid captured! Press 'c' again to capture a new grid.")
        elif key == ord('r'):
            # Reset captured grid
            captured_grid = None
            captured_prob_grid = None
            print("Captured grid reset.")

    cap.stop()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
