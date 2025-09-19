
import cv2 as cv
import numpy as np





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
                    # Preprocess the cell using the existing preprocessing method
                    processed_cell = self.preprocess_cell(cell_image)
                    cells.append(processed_cell)
                else:
                    cells.append(None)
        
        return cells

    def preprocess_cell(self, cell):
        """Preprocess a single cell using the new process_cells method"""
        return self.process_cells(cell)

    def remove_grid_borders(self, cell_image, debug=False):
        """
        Remove grid borders from a cell image by iteratively clearing black pixels from the edges,
        and return the cleaned image (digit only, background white).
        Steps:
        1. Darken the image by 30%.
        2. Blur the image.
        3. Binarize the image (background white, digit black) using Otsu's thresholding.
        4. For each edge, clear strips as long as the black percent is above threshold.
        5. Set those strips to white in the original image.
        6. Binarize the final cleaned image and return it (digit black, background white).
        """
        blur_ksize = 3
        max_val = 255
        black_percent_threshold = 0.35
        edge_strip_width = 3
        max_strips = 35

        img = cell_image.copy()
        img_blur = cv.GaussianBlur(img, (17, 17), 0)
        _, img_bin = cv.threshold(img_blur, 0, max_val, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        img_bin = cv.bitwise_not(img_bin)

        h, w = img_bin.shape
        cleaned = cell_image.copy()

        for side in ['top', 'bottom', 'left', 'right']:
            for i in range(max_strips):
                if side == 'top':
                    y1, y2 = i * edge_strip_width, min((i + 1) * edge_strip_width, h)
                    x1, x2 = 0, w
                elif side == 'bottom':
                    y1, y2 = h - (i + 1) * edge_strip_width, h - i * edge_strip_width
                    y1 = max(y1, 0)
                    x1, x2 = 0, w
                elif side == 'left':
                    x1, x2 = i * edge_strip_width, min((i + 1) * edge_strip_width, w)
                    y1, y2 = 0, h
                elif side == 'right':
                    x1, x2 = w - (i + 1) * edge_strip_width, w - i * edge_strip_width
                    x1 = max(x1, 0)
                    y1, y2 = 0, h

                roi = img_bin[y1:y2, x1:x2]
                total = roi.size
                black = np.count_nonzero(roi == 0)
                black_percent = black / total if total > 0 else 0

                if black_percent > black_percent_threshold:
                    cleaned[y1:y2, x1:x2] = 255
                else:
                    break

        if debug:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 5, figsize=(20, 4))
            axs[0].imshow(cell_image, cmap='gray')
            axs[0].set_title('Original')
            axs[1].imshow(img_blur, cmap='gray')
            axs[1].set_title('Blurred')
            axs[2].imshow(img_bin, cmap='gray')
            axs[2].set_title('Binarized')
            axs[3].imshow(cleaned, cmap='gray')
            axs[3].set_title('Cleaned (White Strips)')
            for ax in axs:
                ax.axis('off')
            plt.show()

        return cleaned

    def process_cells(self, cell_image, debug=False):
        """
        Process cell image to remove grid borders, clean up the image, and sharpen the digit
        """
        cleaned = self.remove_grid_borders(cell_image, debug=debug)

        if len(cleaned.shape) == 3:
            gray = cv.cvtColor(cleaned, cv.COLOR_BGR2GRAY)
        else:
            gray = cleaned.copy()
        
        # Light denoising
        cleaned = cv.medianBlur(gray, 3)
        
        # Enhance contrast
        cleaned = cv.convertScaleAbs(cleaned, alpha=1.3, beta=0)
        
        # Binary threshold (solid digit)
        _, binary = cv.threshold(cleaned, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        
        return binary
        
    
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

