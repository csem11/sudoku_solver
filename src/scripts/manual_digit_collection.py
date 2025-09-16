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
from src.utils.naming import generate_manual_filename


class ManualDigitCollector:
    def __init__(self):
        self.grid = None
        self.cells = None
        self.current_cell = 0
        self.digits = [None] * 81  # 9x9 grid
        self.cell_size = 50
        self.output_size = 450
        self.grid_number = self._get_next_grid_number()
    
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
        
    def capture_grid(self):
        """Capture a Sudoku grid from camera"""
        print("=== GRID CAPTURE ===")
        print("Instructions:")
        print("- Position the Sudoku grid in the camera view")
        print("- Press 'c' to capture when grid is detected")
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
                
                grid_detector = GridDetector(processed)
                grid_contours = grid_detector.find_grid()
                
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
                        
                        # Extract cells using existing module
                        cell_extractor = CellExtractor(self.grid)
                        self.cells = cell_extractor.extract_cells(self.cell_size)
                        
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
                elif key == ord('c') and self.grid is not None:
                    print("Grid captured successfully!")
                    return True
        
        cv.destroyAllWindows()
        return False
    
    def label_digits(self):
        """Navigate through cells and assign digit labels"""
        if self.cells is None:
            print("No grid captured. Please capture a grid first.")
            return
        
        print("\n=== DIGIT LABELING ===")
        print("Navigate through all 81 cells and assign digits 0-9")
        print("All cells must be assigned before saving")
        
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
                    continue
                print("All 81 cells have been assigned digits! Saving...")
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
            elif key == ord('a'):  # Previous cell
                if self.current_cell > 0:
                    self.current_cell -= 1
                    print(f"Moved to cell {self.current_cell}")
                else:
                    print("Already at first cell")
            elif key == ord('d'):  # Next cell
                if self.current_cell < 80:
                    self.current_cell += 1
                    print(f"Moved to cell {self.current_cell}")
                else:
                    print("Already at last cell")
            elif key == ord('w'):  # Up (previous row)
                if self.current_cell >= 9:
                    self.current_cell -= 9
                    print(f"Moved to cell {self.current_cell}")
                else:
                    print("Already at top row")
            elif key == ord('s'):  # Down (next row)
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
        print("- Use arrow keys to navigate between cells")
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
            elif key == 83:  # Right arrow
                self.current_cell = min(80, self.current_cell + 1)
            elif key == 81:  # Left arrow
                self.current_cell = max(0, self.current_cell - 1)
            elif key == 84:  # Down arrow
                self.current_cell = min(80, self.current_cell + 9)
            elif key == 82:  # Up arrow
                self.current_cell = max(0, self.current_cell - 9)
        
        return True
    
    def collect_digits(self):
        """Interactive digit collection for each cell across multiple grids"""
        if not self.grids:
            print("No grids captured. Please capture grids first.")
            return
        
        print(f"\nStarting digit collection for {len(self.grids)} grid(s)...")
        print("Instructions:")
        print("- Use arrow keys to navigate between cells")
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
            elif key == 83:  # Right arrow
                self.current_cell = min(80, self.current_cell + 1)
            elif key == 81:  # Left arrow
                self.current_cell = max(0, self.current_cell - 1)
            elif key == 84:  # Down arrow
                self.current_cell = min(80, self.current_cell + 9)
            elif key == 82:  # Up arrow
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
        """Display the current cell being edited"""
        if self.cells and self.cells[self.current_cell] is not None:
            # Display raw cell image
            raw_cell = self.cells[self.current_cell]
            
            # Resize for better visibility
            display_cell = cv.resize(raw_cell, (200, 200))
            
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
            
            cv.imshow("Current Cell", display_cell)
    
    def _display_current_cell_with_commands(self):
        """Display the current cell with commands overlay"""
        if self.cells and self.cells[self.current_cell] is not None:
            # Display raw cell image
            raw_cell = self.cells[self.current_cell]
            
            # Create a larger display area to fit commands
            display_cell = cv.resize(raw_cell, (500, 500))
            
            # Add current cell info
            row = self.current_cell // 9
            col = self.current_cell % 9
            current_digit = self.digits[self.current_cell]
            
            # Cell info
            info_text = f"Cell {self.current_cell}/80 (Row {row}, Col {col})"
            if current_digit is not None:
                info_text += f" - Digit: {current_digit}"
            else:
                info_text += " - Empty"
            
            # Draw background rectangle for text
            cv.rectangle(display_cell, (5, 5), (495, 35), (0, 0, 0), -1)
            cv.putText(display_cell, info_text, (10, 25), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Commands overlay with background
            commands = [
                "COMMANDS:",
                "0-9: Assign digit",
                "A/D: Left/Right", 
                "W/S: Up/Down",
                "F: Finish & Save",
                "Q: Quit"
            ]
            
            # Draw background for commands
            cv.rectangle(display_cell, (5, 40), (495, 200), (0, 0, 0), -1)
            
            y_offset = 60
            for i, cmd in enumerate(commands):
                color = (0, 255, 0) if i == 0 else (255, 255, 255)
                thickness = 2 if i == 0 else 1
                cv.putText(display_cell, cmd, (10, y_offset + i * 25), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            
            cv.imshow("Current Cell", display_cell)
    
    def _display_grid_overview(self):
        """Display a small overview of the entire grid with current position highlighted"""
        if self.grid is None:
            return
        
        # Create a small overview
        overview = cv.resize(self.grid, (300, 300))
        
        # Highlight current cell
        row = self.current_cell // 9
        col = self.current_cell % 9
        
        cell_width = 300 // 9
        cell_height = 300 // 9
        
        x1 = col * cell_width
        y1 = row * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height
        
        cv.rectangle(overview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add digit labels
        for i, digit in enumerate(self.digits):
            if digit is not None:
                r = i // 9
                c = i % 9
                text_x = c * cell_width + cell_width // 2 - 5
                text_y = r * cell_height + cell_height // 2 + 5
                cv.putText(overview, str(digit), (text_x, text_y), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv.imshow("Grid Overview", overview)
    
    def save_data(self):
        """Save cell images with labels to data directory"""
        # Create output directory
        output_dir = Path("data/digits/manual")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual raw cell images with their assigned digits
        saved_count = 0
        
        for i, (cell, digit) in enumerate(zip(self.cells, self.digits)):
            if cell is not None and digit is not None:
                row = i // 9
                col = i % 9
                
                # Save raw cell image (not processed)
                filename = generate_manual_filename(digit, self.grid_number, i)
                filepath = output_dir / filename
                cv.imwrite(str(filepath), cell)
                saved_count += 1
        
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
        
        # Save summary
        with open(output_dir / "data_summary.txt", "w") as f:
            f.write("Manual Digit Collection Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Grid Number: {self.grid_number}\n")
            f.write(f"Total cells processed: {len([c for c in self.cells if c is not None])}\n")
            f.write(f"Digits assigned: {saved_count}\n")
            f.write(f"Empty cells: {len([d for d in self.digits if d is None])}\n")
            f.write("NOTE: All cells must be assigned digits 0-9 (no blanks allowed)\n")
            f.write("\nDigit distribution:\n")
            for digit in range(10):
                count = sum(1 for d in self.digits if d == digit)
                f.write(f"Digit {digit}: {count} cells\n")
        
        print(f"Saved {saved_count} digit images to {output_dir}/")
        print("Files saved:")
        print(f"- grid_{self.grid_number}.npy (numpy array)")
        print(f"- grid_{self.grid_number}.txt (text format)")
        print("- data_summary.txt (collection summary)")
        print("- Individual digit images with labels and grid numbers in filenames")
        print(f"- Output directory: {output_dir}/")
    
    def _save_digits(self):
        """Save the collected digits to files with labels in filename"""
        # Create output directory
        output_dir = Path("data/digits/manual")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_saved = 0
        all_digits = []
        
        # Process each grid
        for grid_data in self.grids:
            grid_id = grid_data['grid_id']
            grid = grid_data['grid']
            cells = grid_data['cells']
            digits = grid_data['digits']
            
            # Save individual cell images with their assigned digits
            cell_extractor = CellExtractor(grid)
            grid_saved = 0
            
            for i, (cell, digit) in enumerate(zip(cells, digits)):
                if cell is not None and digit is not None:
                    row = i // 9
                    col = i % 9
                    
                    # Use existing preprocessing method
                    processed_cell = cell_extractor.preprocess_cell(cell)
                    
                    # Save with digit label and grid ID in filename
                    filename = generate_manual_filename(digit, grid_id, i)
                    filepath = output_dir / filename
                    cv.imwrite(str(filepath), processed_cell)
                    grid_saved += 1
                    total_saved += 1
            
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
            
            all_digits.extend(digits)
            print(f"Grid {grid_id}: Saved {grid_saved} digit images")
        
        # Save combined summary
        with open(output_dir / "data_summary.txt", "w") as f:
            f.write("Manual Digit Collection Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Total grids processed: {len(self.grids)}\n")
            f.write(f"Total digits assigned: {total_saved}\n")
            f.write(f"Empty cells: {len([d for d in all_digits if d is None])}\n")
            f.write("\nDigit distribution across all grids:\n")
            for digit in range(10):
                count = sum(1 for d in all_digits if d == digit)
                f.write(f"Digit {digit}: {count} cells\n")
        
        print(f"\nSaved {total_saved} digit images from {len(self.grids)} grid(s) to {output_dir}/")
        print("Files saved:")
        print(f"- {len(self.grids)} individual grid files (grid_*.npy, grid_*.txt)")
        print("- data_summary.txt (collection summary)")
        print("- Individual digit images with labels in filenames")
        print(f"- Images saved in: {output_dir}/")
    
    def run(self):
        """Main execution function"""
        print("=== Manual Digit Collection Tool ===")
        print(f"Grid Number: {self.grid_number}")
        
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
