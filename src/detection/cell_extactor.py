from turtle import width
import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract


class CellExtractor:
    def __init__(self, grid):
        self.grid = grid
        self.cells = self.extract_cells()

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

    def preprocess_cell(self, cell):

        cell = self._crop_cell(cell, 10)

        res = cv.resize(cell,None,fx=5, fy=5, interpolation =  cv.INTER_LINEAR)

        gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        # edges = cv.Canny(gray,100,200)
        # Order/repeating seems to help here
        blur = cv.GaussianBlur(gray,(5,5),0)
        # blur = cv.resize(blur,None,fx=2, fy=2, interpolation =  cv.INTER_LINEAR)
        # blur = cv.GaussianBlur(blur,(5,5),0)

        _ ,thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        inverted = cv.bitwise_not(thresh)

        kernel = np.ones((5,5),np.uint8)
        dilation = cv.dilate(inverted,kernel,iterations = 1)
        reinverted = cv.bitwise_not(dilation)
        print(type(reinverted))
        processed_cell = self._resize_cell(reinverted)

        return processed_cell
        
    
    def _crop_cell(self, cell, crop_pct):
        height, width = cell.shape[:2]
        crop_x = int(width * crop_pct / 100)
        crop_y = int(height * crop_pct / 100)
        
        cropped = cell[crop_y:height-crop_y, crop_x:width-crop_x]
    
        return cropped

    def _resize_cell(self, cell, target_size=(28, 28)):

        h, w = cell.shape

        scale = min(target_size[0]/w, target_size[1]/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv.resize(cell, (new_w, new_h))
        
        # Create 28x28 canvas and center the digit
        canvas = np.full(target_size, 255, dtype=np.uint8)
        
        # Calculate position to center
        start_x = (target_size[0] - new_w) // 2
        start_y = (target_size[1] - new_h) // 2
        
        # Place resized image on canvas
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        
        return canvas

    def save_cells(self, cells, output_dir="cells"):
        """Save all extracted cells to files for debugging."""
        import os
        from pathlib import Path
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        print(f"Saving {len(cells)} cells to {output_dir}/")
        
        for i, cell in enumerate(cells):
            if cell is not None:
                # Save as cell_00_row0_col0.jpg, cell_01_row0_col1.jpg, etc.
                row = i // 9
                col = i % 9
                filename = f"{output_dir}/cell_{i:02d}_row{row}_col{col}.jpg"
                clean_cell = self.preprocess_cell(cell)
                cv.imwrite(filename, clean_cell)
                print(f"Saved: {filename}")
            else:
                print(f"Cell {i}: None - not saved")