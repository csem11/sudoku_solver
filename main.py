#!/usr/bin/env python3
"""
Sudoku Solver - Simple grid detection from camera
"""

import cv2 as cv
import numpy as np
from pathlib import Path
from src import Sudoku, VideoCapture, FrameProcessor, GridDetector, GridTransformer, CellExtractor, GridGenerator


def main():
    # Initialize video capture
    cap = VideoCapture(0)
    if not cap.start():
        print("Failed to start video capture.")
        return
    
    frame_processor = FrameProcessor()
    grid_detector = GridDetector()
    grid_transformer = GridTransformer()
    cell_extractor = None
    grid_generator = GridGenerator()

    detected_grid = None
    grid_img = None
    predicted_grid = None
    prob_grid = None

    while True:
        ret, frame = cap.read_frame()
        if not ret:
            print("Failed to capture frame from camera.")
            break

        processed_frame = frame_processor.preprocess(frame)
        grid_contour = grid_detector.detect(processed_frame)

        display_frame = frame.copy()

        if grid_contour is not None:
            # Draw detected grid contour
            cv.drawContours(display_frame, [grid_contour], -1, (0, 255, 0), 2)

            # Transform the grid to a top-down view
            grid_img = grid_transformer.transform(frame, grid_contour)

            # Show the warped grid image
            cv.imshow("Grid Image", grid_img)

            # Extract cells from the grid image
            cell_extractor = CellExtractor(grid_img)
            cells = np.array(cell_extractor.processed_cells)
            # Ensure cells are in the right shape for the generator
            cells = np.array([cell for cell in cells if cell is not None])
            if len(cells) == 81:
                cells = cells.reshape(9, 9, 28, 28)
                cells = cells.reshape(81, 28, 28, 1)
                # Predict the grid
                predicted_grid, prob_grid = grid_generator.generate_grid(cells, include_prob_grid=True)
        else:
            cv.destroyWindow("Grid Image")
            grid_img = None
            predicted_grid = None
            prob_grid = None

        # Show the frame with detected grid
        cv.imshow("Sudoku Detection", display_frame)

        # If we have a predicted grid, show it
        if predicted_grid is not None:
            print("Predicted Grid:")
            print(predicted_grid)
            # Optionally, visualize the board with probabilities
            grid_generator.show_board(predicted_grid, prob_grid)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.stop()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
