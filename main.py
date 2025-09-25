#!/usr/bin/env python3
"""
Sudoku Solver - Real-time grid detection with digit prediction overlay
"""

import cv2 as cv
import numpy as np
import time
from pathlib import Path
from src import VideoCapture, FrameProcessor, GridDetector, GridTransformer, CellExtractor, GridGenerator
from src.generation.sudoku import Sudoku


def overlay_predicted_digits(frame, grid_corners, predicted_grid, prob_grid, cell_size=50):
    """
    Optimized overlay function for predicted digits on the live video frame.
    """
    if grid_corners is None or predicted_grid is None:
        return frame
    
    # Pre-calculate grid dimensions
    x_coords = grid_corners[:, 0]
    y_coords = grid_corners[:, 1]
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    
    cell_width = (max_x - min_x) / 9
    cell_height = (max_y - min_y) / 9
    
    # Pre-calculate font scale and thickness
    font_scale = min(cell_width, cell_height) / 50
    thickness = max(1, int(font_scale * 2))
    radius = int(min(cell_width, cell_height) * 0.3)
    
    # Draw only non-zero digits to reduce processing
    for row in range(9):
        for col in range(9):
            digit = predicted_grid[row, col]
            if digit != 0:  # Skip zero digits
                confidence = prob_grid[row, col] if prob_grid is not None else 1.0
                
                # Calculate cell position
                cell_x = int(min_x + col * cell_width + cell_width / 2)
                cell_y = int(min_y + row * cell_height + cell_height / 2)
                
                # Determine color based on confidence (simplified)
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green
                elif confidence > 0.5:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                # Draw background circle and digit in one pass
                cv.circle(frame, (cell_x, cell_y), radius, (255, 255, 255), -1)
                cv.circle(frame, (cell_x, cell_y), radius, color, 2)
                
                # Draw digit text
                text_size, _ = cv.getTextSize(str(digit), cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = cell_x - text_size[0] // 2
                text_y = cell_y + text_size[1] // 2
                
                cv.putText(frame, str(digit), (text_x, text_y), 
                          cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    
    return frame


def add_confidence_overlay(frame, grid_corners, prob_grid):
    """
    Add confidence statistics overlay above the detected grid.
    
    Args:
        frame: The video frame to overlay on
        grid_corners: The 4 corners of the detected grid
        prob_grid: 9x9 array of confidence scores
        
    Returns:
        frame with confidence overlay
    """
    if grid_corners is None or prob_grid is None:
        return frame
    
    # Calculate grid position
    x_coords = grid_corners[:, 0]
    y_coords = grid_corners[:, 1]
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    
    # Calculate confidence statistics
    non_zero_probs = prob_grid[prob_grid > 0]  # Only consider non-zero predictions
    if len(non_zero_probs) > 0:
        avg_confidence = np.mean(non_zero_probs)
        max_confidence = np.max(non_zero_probs)
        min_confidence = np.min(non_zero_probs)
        high_conf_count = np.sum(non_zero_probs >= 0.8)
        total_predictions = len(non_zero_probs)
    else:
        avg_confidence = 0.0
        max_confidence = 0.0
        min_confidence = 0.0
        high_conf_count = 0
        total_predictions = 0
    
    # Position overlay at fixed location on frame perimeter (top-right corner)
    frame_height, frame_width = frame.shape[:2]
    overlay_width = 250
    overlay_height = 80
    overlay_x = frame_width - overlay_width - 10  # 10 pixels from right edge
    overlay_y = 10  # 10 pixels from top edge
    
    # Draw background rectangle
    cv.rectangle(frame, (overlay_x, overlay_y), 
                (overlay_x + overlay_width, overlay_y + overlay_height), 
                (0, 0, 0), -1)
    cv.rectangle(frame, (overlay_x, overlay_y), 
                (overlay_x + overlay_width, overlay_y + overlay_height), 
                (255, 255, 255), 2)
    
    # Prepare text lines - more compact layout
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    line_height = 12
    text_x = overlay_x + 8
    text_y = overlay_y + 15
    
    # Title
    cv.putText(frame, "Confidence", (text_x, text_y), 
              font, font_scale + 0.1, (255, 255, 255), thickness + 1)
    text_y += line_height + 2
    
    # Statistics - more compact
    solve_threshold = 0.999
    can_solve = avg_confidence >= solve_threshold
    solve_status = "READY" if can_solve else "WAITING"
    solve_color = (0, 255, 0) if can_solve else (0, 0, 255)
    
    stats_lines = [
        f"Avg: {avg_confidence:.3f}",
        f"Max: {max_confidence:.3f}",
        f"High: {high_conf_count}/{total_predictions}",
        f"Solve: {solve_status}"
    ]
    
    for line in stats_lines:
        cv.putText(frame, line, (text_x, text_y), 
                  font, font_scale, (255, 255, 255), thickness)
        text_y += line_height
    
    # Add confidence bar - compact version
    bar_x = text_x
    bar_y = text_y + 3
    bar_width = 120
    bar_height = 8
    
    # Draw background bar
    cv.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                (50, 50, 50), -1)
    
    # Draw confidence bar
    if avg_confidence > 0:
        bar_fill_width = int(bar_width * avg_confidence)
        # Color based on solve readiness
        if can_solve:
            bar_color = (0, 255, 0)  # Green for ready to solve
        elif avg_confidence > 0.8:
            bar_color = (0, 255, 255)  # Yellow for high confidence
        elif avg_confidence > 0.5:
            bar_color = (255, 165, 0)  # Orange for medium confidence
        else:
            bar_color = (0, 0, 255)  # Red for low confidence
        
        cv.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill_width, bar_y + bar_height), 
                    bar_color, -1)
        
        # Draw solve threshold line
        threshold_x = bar_x + int(bar_width * solve_threshold)
        cv.line(frame, (threshold_x, bar_y - 2), (threshold_x, bar_y + bar_height + 2), 
               (255, 255, 255), 2)
    
    # Draw bar border
    cv.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                (255, 255, 255), 1)
    
    return frame


def solve_sudoku_puzzle(puzzle_grid):
    """
    Solve a Sudoku puzzle using the Sudoku class.
    
    Args:
        puzzle_grid: 9x9 numpy array of the puzzle (0 for empty cells)
        
    Returns:
        solved_grid: 9x9 numpy array of the solution, or None if unsolvable
    """
    try:
        # Create a Sudoku instance with the detected puzzle
        sudoku_solver = Sudoku(board=puzzle_grid.copy())
        
        # Attempt to solve the puzzle
        if sudoku_solver.solve_board():
            return sudoku_solver.board
        else:
            print("Warning: Puzzle could not be solved")
            return None
            
    except Exception as e:
        print(f"Error solving puzzle: {e}")
        return None


def overlay_solved_digits(frame, grid_corners, original_grid, solved_grid, prob_grid=None):
    """
    Overlay both original and solved digits on the live video frame.
    Original digits in one color, solved digits in another.
    """
    if grid_corners is None or original_grid is None:
        return frame
    
    # Pre-calculate grid dimensions
    x_coords = grid_corners[:, 0]
    y_coords = grid_corners[:, 1]
    min_x, max_x = int(min(x_coords)), int(max(x_coords))
    min_y, max_y = int(min(y_coords)), int(max(y_coords))
    
    cell_width = (max_x - min_x) / 9
    cell_height = (max_y - min_y) / 9
    
    # Pre-calculate font scale and thickness
    font_scale = min(cell_width, cell_height) / 50
    thickness = max(1, int(font_scale * 2))
    radius = int(min(cell_width, cell_height) * 0.3)
    
    # Draw digits for each cell
    for row in range(9):
        for col in range(9):
            original_digit = original_grid[row, col]
            solved_digit = solved_grid[row, col] if solved_grid is not None else original_digit
            
            # Calculate cell position
            cell_x = int(min_x + col * cell_width + cell_width / 2)
            cell_y = int(min_y + row * cell_height + cell_height / 2)
            
            # Determine what to show
            if original_digit != 0:
                # Show original digit with confidence-based coloring
                digit = original_digit
                confidence = prob_grid[row, col] if prob_grid is not None else 1.0
                
                if confidence > 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                elif confidence > 0.5:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence
                
                # Draw background circle and digit
                cv.circle(frame, (cell_x, cell_y), radius, (255, 255, 255), -1)
                cv.circle(frame, (cell_x, cell_y), radius, color, 2)
                
                # Draw digit text
                text_size, _ = cv.getTextSize(str(digit), cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = cell_x - text_size[0] // 2
                text_y = cell_y + text_size[1] // 2
                
                cv.putText(frame, str(digit), (text_x, text_y), 
                          cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
            
            elif solved_digit != 0 and solved_grid is not None:
                # Show solved digit in blue for empty cells
                digit = solved_digit
                color = (255, 0, 0)  # Blue for solved digits
                
                # Draw background circle and digit
                cv.circle(frame, (cell_x, cell_y), radius, (255, 255, 255), -1)
                cv.circle(frame, (cell_x, cell_y), radius, color, 2)
                
                # Draw digit text
                text_size, _ = cv.getTextSize(str(digit), cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = cell_x - text_size[0] // 2
                text_y = cell_y + text_size[1] // 2
                
                cv.putText(frame, str(digit), (text_x, text_y), 
                          cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    
    return frame


def main():
    # Initialize video capture
    cap = VideoCapture(0)
    if not cap.start():
        return
    
    # Use optimized frame processor with smaller blur kernel
    frame_processor = FrameProcessor(blur_kernel=3)  # Smaller kernel for speed
    grid_detector = GridDetector()
    grid_transformer = GridTransformer(450)
    grid_generator = GridGenerator("models/digit_classifier_10digits.keras")

    predicted_grid = None
    prob_grid = None
    solved_grid = None
    last_grid_corners = None
    show_solution = False  # Toggle for showing solved puzzle
    
    # Performance optimization variables
    last_prediction_time = 0
    prediction_interval = 1.0  # Predict every 1 second max
    cached_cells = None
    cached_grid_corners = None
    last_grid_area = 0
    min_grid_area = 10000  # Minimum grid area to process
    last_preprocess_time = 0
    preprocess_interval = 0.1  # Re-preprocess every 100ms max

    while True:
        ret, frame = cap.read_frame()
        if not ret:
            break

        # Process every frame - no skipping
        # Use cached preprocessing if grid hasn't moved much
        current_time = time.time()
        should_reprocess = (cached_grid_corners is None or 
                           current_time - last_preprocess_time > preprocess_interval or
                           predicted_grid is None)
        
        if should_reprocess:
            processed_frame = frame_processor.preprocess(frame)
            grid_contour = grid_detector.detect(processed_frame)
            last_preprocess_time = current_time
        else:
            # Use cached grid detection for smooth tracking
            grid_contour = None
            if cached_grid_corners is not None:
                # Create a dummy contour from cached corners for overlay positioning
                grid_contour = cached_grid_corners.reshape(-1, 1, 2).astype(np.int32)

        display_frame = frame.copy()

        if grid_contour is not None:
            # Calculate grid area for size filtering
            grid_area = cv.contourArea(grid_contour)
            
            # Skip processing if grid is too small
            if grid_area < min_grid_area:
                if predicted_grid is not None and last_grid_corners is not None:
                    display_frame = overlay_predicted_digits(
                        display_frame, last_grid_corners, predicted_grid, prob_grid
                    )
                cv.imshow("Sudoku Solver", display_frame)
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                continue
            
            # Extract grid corners for overlay positioning
            perimeter = cv.arcLength(grid_contour, True)
            epsilon = 0.02 * perimeter
            grid_corners = cv.approxPolyDP(grid_contour, epsilon, True)
            
            if len(grid_corners) == 4:
                grid_corners_reshaped = grid_corners.reshape(4, 2)
                
                # Movement detection for prediction throttling
                grid_moved = False
                area_changed = False
                if cached_grid_corners is not None:
                    # Calculate movement distance
                    movement = np.linalg.norm(grid_corners_reshaped - cached_grid_corners)
                    grid_moved = movement > 30  # Moderate threshold
                    area_changed = abs(grid_area - last_grid_area) > 3000
                
                should_predict = (current_time - last_prediction_time > prediction_interval or 
                                grid_moved or area_changed or cached_cells is None)
                
                if should_predict:
                    try:
                        # Transform the grid to a top-down view
                        grid_img = grid_transformer.transform(frame, grid_contour)

                        # Extract cells from the grid image
                        cell_extractor = CellExtractor(grid_img)
                        cells = np.array(cell_extractor.processed_cells)
                        cells = np.array([cell for cell in cells if cell is not None])
                        
                        if len(cells) == 81:
                            cells = cells.reshape(9, 9, 28, 28)
                            cells = cells.reshape(81, 28, 28, 1)
                            
                            # Cache the cells and corners
                            cached_cells = cells
                            cached_grid_corners = grid_corners_reshaped.copy()
                            last_prediction_time = current_time
                            last_grid_area = grid_area
                            
                            # Perform digit prediction
                            predicted_grid, prob_grid = grid_generator.generate_grid(cells, include_prob_grid=True)
                            
                            # Only attempt to solve if confidence is very high (>= 99.9%)
                            if predicted_grid is not None and prob_grid is not None:
                                # Calculate average confidence of non-zero predictions
                                non_zero_probs = prob_grid[prob_grid > 0]
                                if len(non_zero_probs) > 0:
                                    avg_confidence = np.mean(non_zero_probs)
                                    
                                    if avg_confidence >= 0.999:  # 99.9% confidence threshold
                                        print(f"High confidence detected ({avg_confidence:.3f}), attempting to solve...")
                                        solved_grid = solve_sudoku_puzzle(predicted_grid)
                                        if solved_grid is not None:
                                            print("Puzzle solved successfully!")
                                        else:
                                            print("Could not solve puzzle")
                                    else:
                                        print(f"Confidence too low ({avg_confidence:.3f}), skipping solve")
                                        solved_grid = None
                                else:
                                    print("No predictions made, skipping solve")
                                    solved_grid = None
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        pass
                
                # Update corners for overlay positioning (every frame)
                last_grid_corners = grid_corners_reshaped
                
                # Overlay digits on the live video frame
                if predicted_grid is not None:
                    if show_solution and solved_grid is not None:
                        # Show solved puzzle with both original and solved digits
                        display_frame = overlay_solved_digits(
                            display_frame, last_grid_corners, predicted_grid, solved_grid, prob_grid
                        )
                    else:
                        # Show only original predicted digits
                        display_frame = overlay_predicted_digits(
                            display_frame, last_grid_corners, predicted_grid, prob_grid
                        )
        else:
            predicted_grid = None
            prob_grid = None
            solved_grid = None
            last_grid_corners = None
            cached_cells = None
            cached_grid_corners = None

        # Add status text to frame
        status_text = f"Solution: {'ON' if show_solution else 'OFF'}"
        cv.putText(display_frame, status_text, (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(display_frame, status_text, (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Add confidence overlay above the grid
        if predicted_grid is not None and prob_grid is not None and last_grid_corners is not None:
            display_frame = add_confidence_overlay(display_frame, last_grid_corners, prob_grid)

        # Show the frame with predicted digits
        cv.imshow("Sudoku Solver", display_frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Toggle solution display
            show_solution = not show_solution
            print(f"Solution display: {'ON' if show_solution else 'OFF'}")

    cap.stop()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
