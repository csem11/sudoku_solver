#!/usr/bin/env python3
"""
Manual Digit Collection Tool
Captures a Sudoku grid and allows manual digit assignment for each cell
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.capture import VideoCapture, FrameProcessor
from src.detection import GridDetector, GridTransformer, CellExtractor
from src.generation import GridGenerator
from src.utils.naming import generate_manual_filename


class ManualDigitCollector:
    def __init__(self):
        self.grid = None
        self.cells = None
        self.raw_cells = None  # Store raw cell images
        self.processed_cells = None  # Store processed cell images
        self.current_cell = 0
        self.digits = [None] * 81  # 9x9 grid
        self.predicted_digits = [None] * 81  # AI predictions
        self.prediction_confidence = [0.0] * 81  # Confidence scores
        self.cell_size = 50
        self.output_size = 450
        self.grid_number = self._get_next_grid_number()
        self.grid_generator = GridGenerator()
        self.use_predictions = True  # Flag to enable/disable predictions
    
    def _get_next_grid_number(self):
        """Get next available grid number to avoid overwriting data"""
        output_dir = Path("data/digits/manual")
        if not output_dir.exists():
            return 0
        
        # Find existing grid files
        existing_grids = []
        for item in output_dir.iterdir():
            if item.is_file() and item.name.startswith("grid_"):
                try:
                    grid_num = int(item.name.split("_")[1].split(".")[0])
                    existing_grids.append(grid_num)
                except (ValueError, IndexError):
                    continue
        
        return max(existing_grids, default=-1) + 1
    
    def _generate_predictions(self):
        """Generate AI predictions for all cells"""
        if self.processed_cells is None or len(self.processed_cells) != 81:
            print("Cannot generate predictions: need 81 processed cells")
            return False
        
        try:
            # Convert processed cells to the format expected by GridGenerator
            cell_images = []
            for cell in self.processed_cells:
                if cell is not None:
                    # Ensure cell is in the right format (28x28 grayscale)
                    if len(cell.shape) == 2:
                        cell = cell.reshape(28, 28, 1)
                    elif len(cell.shape) == 3 and cell.shape[2] == 1:
                        # Already in correct format
                        pass
                    else:
                        # Convert to grayscale and reshape
                        if len(cell.shape) == 3:
                            cell = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
                        cell = cell.reshape(28, 28, 1)
                    cell_images.append(cell)
                else:
                    # Create empty cell for missing cells
                    empty_cell = np.zeros((28, 28, 1), dtype=np.uint8)
                    cell_images.append(empty_cell)
            
            cell_images = np.array(cell_images)
            
            # Generate predictions
            predicted_grid, prob_grid = self.grid_generator.generate_grid(cell_images, include_prob_grid=True)
            
            # Flatten to 1D arrays
            self.predicted_digits = predicted_grid.flatten().tolist()
            self.prediction_confidence = prob_grid.flatten().tolist()
            
            # Pre-populate digits with predictions
            self.digits = self.predicted_digits.copy()
            
            print(f"Generated predictions for {len([d for d in self.predicted_digits if d != 0])} non-empty cells")
            print(f"Average confidence: {np.mean(self.prediction_confidence):.3f}")
            
            return True
            
        except Exception as e:
            print(f"Error generating predictions: {e}")
            return False
        
    def capture_grid(self):
        """Capture a Sudoku grid from camera"""
        print("=== GRID CAPTURE ===")
        print("Instructions:")
        print("- Position the Sudoku grid in the camera view")
        print("- Press 'c' to capture when grid is detected")
        print("- Press 'p' to toggle AI predictions on/off")
        print("- Press 'q' to quit")
        
        frame_processor = FrameProcessor()
        
        with VideoCapture() as video:
            while True:
                success, frame = video.read_frame()
                if not success:
                    print("Failed to read frame")
                    break
                
                # Process frame to detect grid
                processed = frame_processor.preprocess(frame)
                display_frame = frame.copy()
                
                grid_detector = GridDetector()
                grid_contours = grid_detector.detect(processed)
                
                if grid_contours is not None and len(grid_contours) > 0:
                    # Draw grid outline
                    cv.drawContours(display_frame, [grid_contours], -1, (0, 255, 0), 3)
                    
                    # Use the same approach as main.py - this is the standard pattern
                    # for extracting corners from the detected contour
                    perimeter = cv.arcLength(grid_contours, True)
                    epsilon = 0.02 * perimeter
                    approx = cv.approxPolyDP(grid_contours, epsilon, True)
                    
                    if len(approx) == 4:
                        # Transform grid using existing module
                        grid_transformer = GridTransformer(self.output_size)
                        self.grid = grid_transformer.transform(frame, approx)
                        
                        # Extract both raw and processed cells using existing module
                        cell_extractor = CellExtractor(self.grid)
                        # Use the new attribute structure
                        self.raw_cells = cell_extractor.raw_cells
                        self.processed_cells = cell_extractor.processed_cells
                        
                        # Keep cells for backward compatibility (processed cells)
                        self.cells = self.processed_cells
                        
                        cv.putText(display_frame, "Grid detected! Press 'c' to capture", 
                                 (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv.putText(display_frame, "No valid grid found", 
                                 (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv.putText(display_frame, "No grid detected", 
                             (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv.imshow("Grid Capture", display_frame)
                
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    return False
                elif key == ord('p'):  # Toggle predictions
                    self.use_predictions = not self.use_predictions
                    print(f"AI Predictions: {'Enabled' if self.use_predictions else 'Disabled'}")
                elif key == ord('c') and self.grid is not None:
                    print("Grid captured successfully!")
                    
                    # Generate AI predictions if enabled
                    if self.use_predictions:
                        print("Generating AI predictions...")
                        if self._generate_predictions():
                            print("Predictions generated! You can now review and correct them.")
                            self._show_prediction_stats()
                        else:
                            print("Failed to generate predictions. Starting with empty grid.")
                            # Reset to empty if predictions failed
                            self.digits = [None] * 81
                            self.predicted_digits = [None] * 81
                            self.prediction_confidence = [0.0] * 81
                    
                    # Position windows to avoid overlap
                    self._position_windows()
                    return True
        
        cv.destroyAllWindows()
        return False
    
    def _position_windows(self):
        """Position all windows horizontally side by side"""
        # Simple horizontal positioning - no screen size detection needed
        cell_x = 50
        cell_y = 50
        overview_x = 600  # Position overview window to the right of cell window
        
        overview_y = 50
        
        
        # Position windows with a small delay to ensure they exist
        import time
        time.sleep(0.1)  # Small delay to ensure windows are created
        
        try:
            cv.moveWindow("Current Cell (Processed)", cell_x, cell_y)
        except:
            pass  # Window might not exist yet
        
        try:
            cv.moveWindow("Grid Overview", overview_x, overview_y)
        except:
            pass  # Window might not exist yet
    
    def label_digits(self):
        """Navigate through cells and assign digit labels"""
        if self.cells is None or self.processed_cells is None:
            print("No grid captured. Please capture a grid first.")
            return
        
        print("\n=== DIGIT LABELING ===")
        print("Navigate through all 81 processed cells and assign digits 0-9")
        print("You will see the preprocessed cell images (same as training data)")
        print("AI predictions are pre-populated - review and correct as needed")
        print("Use A/D/W/S to navigate, number keys to assign digits")
        print("All cells must be assigned before saving")
        
        # Ensure windows are properly positioned at start
        self._position_windows()
        
        while True:
            # Display current cell with commands
            self._display_current_cell_with_commands()
            
            # Show grid overview
            self._display_grid_overview()
            
            key = cv.waitKey(0) & 0xFF
            
            if key == ord('q'):
                print("Exiting without saving...")
                return False
            elif key == ord('f'):  # Finish/Save
                # Verify all cells are assigned
                unassigned = [i for i, d in enumerate(self.digits) if d is None]
                if unassigned:
                    print(f"ERROR: {len(unassigned)} cells are still unassigned: {unassigned}")
                    print("All cells must be assigned digits 0-9 before saving.")
                    print("Press 'i' to see prediction statistics for guidance.")
                    continue
                
                # Show final statistics
                print("All 81 cells have been assigned digits!")
                if any(self.predicted_digits):
                    print("\nFinal Statistics:")
                    self._show_prediction_stats()
                    
                    # Show correction summary
                    corrections = sum(1 for i in range(81) 
                                    if self.digits[i] != self.predicted_digits[i] 
                                    and self.predicted_digits[i] is not None)
                    print(f"Corrections made: {corrections}")
                
                print("Saving...")
                break
            elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), 
                        ord('6'), ord('7'), ord('8'), ord('9')]:
                digit = int(chr(key))
                self.digits[self.current_cell] = digit
                print(f"Cell {self.current_cell}: Assigned digit {digit}")
                # Auto-advance to next cell after assignment
                if self.current_cell < 80:
                    self.current_cell += 1
                    print(f"Auto-advanced to cell {self.current_cell}")
                else:
                    print("Reached last cell - use A/W to go back")
            elif key == ord('r'):  # Reset to AI prediction
                if self.predicted_digits[self.current_cell] is not None:
                    self.digits[self.current_cell] = self.predicted_digits[self.current_cell]
                    print(f"Cell {self.current_cell}: Reset to AI prediction {self.predicted_digits[self.current_cell]}")
                else:
                    print(f"Cell {self.current_cell}: No AI prediction available")
            elif key == ord('i'):  # Show prediction info
                self._show_prediction_stats()
            elif key == ord('a'):  # Left
                if self.current_cell > 0:
                    self.current_cell -= 1
                    print(f"Moved to cell {self.current_cell}")
                else:
                    print("Already at first cell")
            elif key == ord('d'):  # Right
                if self.current_cell < 80:
                    self.current_cell += 1
                    print(f"Moved to cell {self.current_cell}")
                else:
                    print("Already at last cell")
            elif key == ord('w'):  # Up
                if self.current_cell >= 9:
                    self.current_cell -= 9
                    print(f"Moved to cell {self.current_cell}")
                else:
                    print("Already at top row")
            elif key == ord('s'):  # Down
                if self.current_cell < 72:
                    self.current_cell += 9
                    print(f"Moved to cell {self.current_cell}")
                else:
                    print("Already at bottom row")
        
        cv.destroyAllWindows()
        return True
    
    def _assign_digits_for_current_grid(self, grid_index):
        """Allow digit assignment for a specific grid immediately after capture"""
        print(f"\nAssigning digits for Grid {grid_index + 1}...")
        print("Instructions:")
        print("- Use A/D/W/S to navigate between cells")
        print("- Press number keys (0-9) to assign digits")
        print("- Press 'x' to clear current cell")
        print("- Press 'd' when done with this grid")
        print("- Press 'q' to quit without saving")
        
        # Set current grid
        self._set_current_grid(grid_index)
        
        while True:
            # Display current cell
            self._display_current_cell()
            
            # Show grid overview
            self._display_grid_overview()
            
            key = cv.waitKey(0) & 0xFF
            
            if key == ord('q'):
                print("Exiting without saving...")
                return False
            elif key == ord('d'):  # Done with this grid
                print(f"Finished assigning digits for Grid {grid_index + 1}")
                break
            elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), 
                        ord('6'), ord('7'), ord('8'), ord('9')]:
                digit = int(chr(key))
                self.current_digits[self.current_cell] = digit
                # Update the grid data with the assigned digit
                self.grids[grid_index]['digits'][self.current_cell] = digit
                print(f"Grid {grid_index + 1}, Cell {self.current_cell}: Assigned digit {digit}")
            elif key == ord('x'):  # Clear current cell
                self.current_digits[self.current_cell] = None
                # Update the grid data to clear the digit
                self.grids[grid_index]['digits'][self.current_cell] = None
                print(f"Grid {grid_index + 1}, Cell {self.current_cell}: Cleared")
            elif key == ord('d'):  # Right
                self.current_cell = min(80, self.current_cell + 1)
            elif key == ord('a'):  # Left
                self.current_cell = max(0, self.current_cell - 1)
            elif key == ord('s'):  # Down
                self.current_cell = min(80, self.current_cell + 9)
            elif key == ord('w'):  # Up
                self.current_cell = max(0, self.current_cell - 9)
        
        return True
    
    def collect_digits(self):
        """Interactive digit collection for each cell across multiple grids"""
        if not self.grids:
            print("No grids captured. Please capture grids first.")
            return
        
        print(f"\nStarting digit collection for {len(self.grids)} grid(s)...")
        print("Instructions:")
        print("- Use A/D/W/S to navigate between cells")
        print("- Press number keys (0-9) to assign digits")
        print("- Press 'x' to clear current cell")
        print("- Press 'n' to go to next grid")
        print("- Press 'p' to go to previous grid")
        print("- Press 's' to save and exit")
        print("- Press 'q' to quit without saving")
        
        current_grid_index = 0
        self._set_current_grid(current_grid_index)
        
        while True:
            # Display current cell
            self._display_current_cell()
            
            # Show grid overview
            self._display_grid_overview()
            
            key = cv.waitKey(0) & 0xFF
            
            if key == ord('q'):
                print("Exiting without saving...")
                break
            elif key == ord('s'):
                self._save_digits()
                break
            elif key == 27:  # Escape key
                break
            elif key == ord('n'):  # Next grid
                current_grid_index = min(len(self.grids) - 1, current_grid_index + 1)
                self._set_current_grid(current_grid_index)
                print(f"Switched to grid {current_grid_index + 1}")
            elif key == ord('p'):  # Previous grid
                current_grid_index = max(0, current_grid_index - 1)
                self._set_current_grid(current_grid_index)
                print(f"Switched to grid {current_grid_index + 1}")
            elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), 
                        ord('6'), ord('7'), ord('8'), ord('9')]:
                digit = int(chr(key))
                self.current_digits[self.current_cell] = digit
                # Update the grid data with the assigned digit
                self.grids[current_grid_index]['digits'][self.current_cell] = digit
                print(f"Grid {current_grid_index + 1}, Cell {self.current_cell}: Assigned digit {digit}")
            elif key == ord('x'):  # Clear current cell
                self.current_digits[self.current_cell] = None
                # Update the grid data to clear the digit
                self.grids[current_grid_index]['digits'][self.current_cell] = None
                print(f"Grid {current_grid_index + 1}, Cell {self.current_cell}: Cleared")
            elif key == ord('d'):  # Right
                self.current_cell = min(80, self.current_cell + 1)
            elif key == ord('a'):  # Left
                self.current_cell = max(0, self.current_cell - 1)
            elif key == ord('s'):  # Down
                self.current_cell = min(80, self.current_cell + 9)
            elif key == ord('w'):  # Up
                self.current_cell = max(0, self.current_cell - 9)
        
        cv.destroyAllWindows()
    
    def _set_current_grid(self, grid_index):
        """Set the current grid for editing"""
        if 0 <= grid_index < len(self.grids):
            self.current_grid = self.grids[grid_index]['grid']
            self.current_cells = self.grids[grid_index]['cells']
            self.current_digits = self.grids[grid_index]['digits']
            self.current_cell = 0  # Reset to first cell
    
    def _display_current_cell(self):
        """Display the current processed cell being edited"""
        if self.processed_cells and self.processed_cells[self.current_cell] is not None:
            # Display processed cell image
            processed_cell = self.processed_cells[self.current_cell]
            
            # Resize for better visibility (processed cells are 28x28, so scale up)
            display_cell = cv.resize(processed_cell, (200, 200), interpolation=cv.INTER_NEAREST)
            
            # Add current cell info
            row = self.current_cell // 9
            col = self.current_cell % 9
            current_digit = self.digits[self.current_cell]
            
            info_text = f"Cell {self.current_cell} (Row {row}, Col {col})"
            if current_digit is not None:
                info_text += f" - Digit: {current_digit}"
            else:
                info_text += " - Empty"
            
            cv.putText(display_cell, info_text, (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv.imshow("Current Cell (Processed)", display_cell)
    
    def _display_current_cell_with_commands(self):
        """Display the current processed cell with commands overlay"""
        if self.processed_cells and self.processed_cells[self.current_cell] is not None:
            # Display processed cell image
            processed_cell = self.processed_cells[self.current_cell]
            
            # Create a larger display area to fit commands (scale up 28x28 to 500x500)
            display_cell = cv.resize(processed_cell, (500, 500), interpolation=cv.INTER_NEAREST)
            
            # Add current cell info
            row = self.current_cell // 9
            col = self.current_cell % 9
            current_digit = self.digits[self.current_cell]
            predicted_digit = self.predicted_digits[self.current_cell] if self.predicted_digits[self.current_cell] is not None else None
            confidence = self.prediction_confidence[self.current_cell] if self.prediction_confidence[self.current_cell] is not None else 0.0
            
            # Cell info
            info_text = f"Cell {self.current_cell}/80 (Row {row}, Col {col})"
            if current_digit is not None:
                info_text += f" - Digit: {current_digit}"
            else:
                info_text += " - Empty"
            
            # Prediction info
            if predicted_digit is not None and predicted_digit != 0:
                info_text += f" [AI: {predicted_digit} ({confidence:.2f})]"
            
            # Draw background rectangle for text with better contrast
            cv.rectangle(display_cell, (5, 5), (495, 40), (0, 0, 0), -1)
            cv.rectangle(display_cell, (7, 7), (493, 38), (50, 50, 50), 2)  # Border for better visibility
            cv.putText(display_cell, info_text, (15, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Commands overlay with background
            commands = [
                "COMMANDS:",
                "0-9: Assign digit",
                "A/D/W/S: Navigate", 
                "R: Reset to AI prediction",
                "I: Show prediction stats",
                "F: Finish & Save",
                "Q: Quit"
            ]
            
            # Draw background for commands with better contrast
            cv.rectangle(display_cell, (5, 45), (495, 240), (0, 0, 0), -1)
            cv.rectangle(display_cell, (7, 47), (493, 238), (50, 50, 50), 2)  # Border for better visibility
            
            y_offset = 70
            for i, cmd in enumerate(commands):
                color = (0, 150, 0) if i == 0 else (255, 255, 255)  # Darker green for header
                thickness = 2 if i == 0 else 1
                font_scale = 0.6 if i == 0 else 0.5
                cv.putText(display_cell, cmd, (15, y_offset + i * 28), 
                          cv.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            
            cv.imshow("Current Cell (Processed)", display_cell)
            # Ensure proper window positioning
            self._position_windows()
    
    def _display_grid_overview(self):
        """Display a small overview of the entire grid with current position highlighted"""
        if self.grid is None:
            return
        
        # Create a small overview
        overview = cv.resize(self.grid, (300, 300))
        
        # Highlight current cell with better visibility
        row = self.current_cell // 9
        col = self.current_cell % 9
        
        cell_width = 300 // 9
        cell_height = 300 // 9
        
        x1 = col * cell_width
        y1 = row * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height
        
        # Draw a thicker, more visible highlight
        cv.rectangle(overview, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # Add a subtle inner border
        cv.rectangle(overview, (x1+2, y1+2), (x2-2, y2-2), (0, 200, 0), 1)
        
        # Add digit labels with confidence-based coloring
        for i, digit in enumerate(self.digits):
            if digit is not None:
                r = i // 9
                c = i % 9
                text_x = c * cell_width + cell_width // 2 - 5
                text_y = r * cell_height + cell_height // 2 + 5
                
                # Color based on confidence if available
                if i < len(self.prediction_confidence) and self.prediction_confidence[i] > 0:
                    conf = self.prediction_confidence[i]
                    # Green for high confidence, yellow for medium, red for low
                    if conf > 0.8:
                        color = (0, 255, 0)  # Green
                    elif conf > 0.5:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 0, 255)  # Red
                else:
                    color = (0, 255, 0)  # Default green
                
                cv.putText(overview, str(digit), (text_x, text_y), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv.imshow("Grid Overview", overview)
        # Ensure proper window positioning
        self._position_windows()
    
    def save_data(self):
        """Save both raw and processed cell images with labels to data directory"""
        # Create output directories
        output_dir = Path("data/digits/manual")
        raw_dir = output_dir / "raw"
        processed_dir = output_dir / "processed"
        grids_dir = Path("data/grids")
        
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        grids_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual raw and processed cell images with their assigned digits
        raw_saved_count = 0
        processed_saved_count = 0
        
        for i, (raw_cell, processed_cell, digit) in enumerate(zip(self.raw_cells, self.processed_cells, self.digits)):
            if raw_cell is not None and processed_cell is not None and digit is not None:
                row = i // 9
                col = i % 9
                
                # Save raw cell image
                raw_filename = generate_manual_filename(digit, self.grid_number, i, "raw")
                raw_filepath = raw_dir / raw_filename
                cv.imwrite(str(raw_filepath), raw_cell)
                raw_saved_count += 1
                
                # Save processed cell image
                processed_filename = generate_manual_filename(digit, self.grid_number, i, "processed")
                processed_filepath = processed_dir / processed_filename
                cv.imwrite(str(processed_filepath), processed_cell)
                processed_saved_count += 1
        
        # Save the complete grid as a numpy array
        grid_array = np.array(self.digits).reshape(9, 9)
        np.save(output_dir / f"grid_{self.grid_number}.npy", grid_array)
        
        # Save as text file for easy reading
        with open(output_dir / f"grid_{self.grid_number}.txt", "w") as f:
            f.write(f"Grid {self.grid_number}\n")
            f.write("=" * 20 + "\n")
            for row in grid_array:
                row_str = " ".join([str(d) if d is not None else "." for d in row])
                f.write(row_str + "\n")
        
        # Save the grid image to data/grids directory
        if self.grid is not None:
            grid_filename = f"grid_{self.grid_number}.jpg"
            grid_filepath = grids_dir / grid_filename
            cv.imwrite(str(grid_filepath), self.grid)
            print(f"Saved grid image to {grid_filepath}")
        
        # Save summary
        with open(output_dir / "data_summary.txt", "w") as f:
            f.write("Manual Digit Collection Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Grid Number: {self.grid_number}\n")
            f.write(f"Total cells processed: {len([c for c in self.cells if c is not None])}\n")
            f.write(f"Raw images saved: {raw_saved_count}\n")
            f.write(f"Processed images saved: {processed_saved_count}\n")
            f.write(f"Empty cells: {len([d for d in self.digits if d is None])}\n")
            f.write("NOTE: All cells must be assigned digits 0-9 (no blanks allowed)\n")
            f.write("\nDigit distribution:\n")
            for digit in range(10):
                count = sum(1 for d in self.digits if d == digit)
                f.write(f"Digit {digit}: {count} cells\n")
        
        print(f"Saved {raw_saved_count} raw digit images to {raw_dir}/")
        print(f"Saved {processed_saved_count} processed digit images to {processed_dir}/")
        print("Files saved:")
        print(f"- grid_{self.grid_number}.npy (numpy array)")
        print(f"- grid_{self.grid_number}.txt (text format)")
        print(f"- grid_{self.grid_number}.jpg (grid image)")
        print("- data_summary.txt (collection summary)")
        print("- Individual raw digit images in raw/ subdirectory")
        print("- Individual processed digit images in processed/ subdirectory")
        print(f"- Digit data directory: {output_dir}/")
        print(f"- Grid image directory: {grids_dir}/")
    
    def _save_digits(self):
        """Save the collected digits to files with labels in filename"""
        # Create output directories
        output_dir = Path("data/digits/manual")
        raw_dir = output_dir / "raw"
        processed_dir = output_dir / "processed"
        grids_dir = Path("data/grids")
        
        raw_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        grids_dir.mkdir(parents=True, exist_ok=True)
        
        total_raw_saved = 0
        total_processed_saved = 0
        all_digits = []
        
        # Process each grid
        for grid_data in self.grids:
            grid_id = grid_data['grid_id']
            grid = grid_data['grid']
            cells = grid_data['cells']
            digits = grid_data['digits']
            
            # Save individual raw and processed cell images with their assigned digits
            cell_extractor = CellExtractor(grid)
            # Use the new attribute structure
            raw_cells = cell_extractor.raw_cells
            processed_cells = cell_extractor.processed_cells
            
            grid_raw_saved = 0
            grid_processed_saved = 0
            
            for i, (raw_cell, processed_cell, digit) in enumerate(zip(raw_cells, processed_cells, digits)):
                if raw_cell is not None and processed_cell is not None and digit is not None:
                    row = i // 9
                    col = i % 9
                    
                    # Save raw cell with digit label and grid ID in filename
                    raw_filename = generate_manual_filename(digit, grid_id, i, "raw")
                    raw_filepath = raw_dir / raw_filename
                    cv.imwrite(str(raw_filepath), raw_cell)
                    grid_raw_saved += 1
                    total_raw_saved += 1
                    
                    # Save processed cell with digit label and grid ID in filename
                    processed_filename = generate_manual_filename(digit, grid_id, i, "processed")
                    processed_filepath = processed_dir / processed_filename
                    cv.imwrite(str(processed_filepath), processed_cell)
                    grid_processed_saved += 1
                    total_processed_saved += 1
            
            # Save individual grid as numpy array
            grid_array = np.array(digits).reshape(9, 9)
            np.save(output_dir / f"grid_{grid_id}.npy", grid_array)
            
            # Save individual grid as text file
            with open(output_dir / f"grid_{grid_id}.txt", "w") as f:
                f.write(f"Grid {grid_id}\n")
                f.write("=" * 20 + "\n")
                for row in grid_array:
                    row_str = " ".join([str(d) if d is not None else "." for d in row])
                    f.write(row_str + "\n")
            
            # Save the grid image to data/grids directory
            if grid is not None:
                grid_filename = f"grid_{grid_id}.jpg"
                grid_filepath = grids_dir / grid_filename
                cv.imwrite(str(grid_filepath), grid)
                print(f"Grid {grid_id}: Saved grid image to {grid_filepath}")
            
            all_digits.extend(digits)
            print(f"Grid {grid_id}: Saved {grid_raw_saved} raw and {grid_processed_saved} processed digit images")
        
        # Save combined summary
        with open(output_dir / "data_summary.txt", "w") as f:
            f.write("Manual Digit Collection Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Total grids processed: {len(self.grids)}\n")
            f.write(f"Total raw images saved: {total_raw_saved}\n")
            f.write(f"Total processed images saved: {total_processed_saved}\n")
            f.write(f"Empty cells: {len([d for d in all_digits if d is None])}\n")
            f.write("\nDigit distribution across all grids:\n")
            for digit in range(10):
                count = sum(1 for d in all_digits if d == digit)
                f.write(f"Digit {digit}: {count} cells\n")
        
        print(f"\nSaved {total_raw_saved} raw and {total_processed_saved} processed digit images from {len(self.grids)} grid(s)")
        print("Files saved:")
        print(f"- {len(self.grids)} individual grid files (grid_*.npy, grid_*.txt)")
        print(f"- {len(self.grids)} grid images (grid_*.jpg)")
        print("- data_summary.txt (collection summary)")
        print("- Individual raw digit images in raw/ subdirectory")
        print("- Individual processed digit images in processed/ subdirectory")
        print(f"- Digit data directory: {output_dir}/")
        print(f"- Grid image directory: {grids_dir}/")
    
    def run(self):
        """Main execution function"""
        print("=== Manual Digit Collection Tool ===")
        print(f"Grid Number: {self.grid_number}")
        print(f"AI Predictions: {'Enabled' if self.use_predictions else 'Disabled'}")
        print("Press 'p' during capture to toggle predictions on/off")
        
        # Step 1: Capture grid
        if not self.capture_grid():
            print("Grid capture failed or cancelled.")
            return
        
        # Step 2: Label digits
        if not self.label_digits():
            print("Digit labeling cancelled.")
            return
        
        # Step 3: Save data
        self.save_data()
        
        print("Manual digit collection completed!")
    
    def _show_prediction_stats(self):
        """Show statistics about AI predictions"""
        if not any(self.predicted_digits):
            print("No AI predictions available")
            return
        
        non_zero_predictions = [d for d in self.predicted_digits if d is not None and d != 0]
        if not non_zero_predictions:
            print("No non-zero predictions made")
            return
        
        avg_confidence = np.mean([self.prediction_confidence[i] for i, d in enumerate(self.predicted_digits) 
                                 if d is not None and d != 0])
        
        print(f"\n=== AI Prediction Statistics ===")
        print(f"Non-zero predictions: {len(non_zero_predictions)}/81")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"High confidence (>0.8): {sum(1 for c in self.prediction_confidence if c > 0.8)}")
        print(f"Medium confidence (0.5-0.8): {sum(1 for c in self.prediction_confidence if 0.5 <= c <= 0.8)}")
        print(f"Low confidence (<0.5): {sum(1 for c in self.prediction_confidence if 0 < c < 0.5)}")


def main():
    """Main function"""
    print("Sudoku Manual Digit Collection Tool")
    print("=" * 40)
    
    try:
        collector = ManualDigitCollector()
        collector.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
