import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract


class CellExtractor:
    def __init__(self, grid):
        self.grid = grid

    def extract_cells(self, cell_size=50):
        cells = []
        
        for row in range(9):
            for col in range(9):
                start_x = col * cell_size
                end_x = start_x + cell_size

                start_y = row * cell_size
                end_y = start_y + cell_size

                # Check bounds
                if (end_x <= self.grid.shape[1] and end_y <= self.grid.shape[0]):
                    cell_image = self.grid[start_y:end_y, start_x:end_x]
                    cells.append(cell_image)
                else:
                    cells.append(None)
        
        return cells

    def get_cell_image():
        pass

    def preprocess_cell():
        pass
        
    def show_sample_cells(self, cells, num_samples=4):
        """Show a few sample cells for debugging."""
        if not cells:
            print("No cells to show")
            return
            
        # Show first few cells
        for i in range(min(num_samples, len(cells))):
            if cells[i] is not None:
                cv.imshow(f"Cell {i}", cells[i])
                print(f"Cell {i} shape: {cells[i].shape}")
            else:
                print(f"Cell {i}: None")