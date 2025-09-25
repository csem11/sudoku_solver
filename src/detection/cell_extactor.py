
import cv2 as cv
import numpy as np





class CellExtractor:
    def __init__(self, grid, debug=False):
        self.original_grid = grid.copy()
        self.grid = self.process_grid(grid, debug=debug)
        self.raw_cells, self.processed_cells = self.extract_raw_and_processed_cells()

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
    
    def extract_raw_and_processed_cells(self, cell_size=50):
        """Extract both raw and processed cells"""
        raw_cells = []
        processed_cells = []
        
        for row in range(9):
            for col in range(9):
                start_x = col * cell_size
                end_x = start_x + cell_size

                start_y = row * cell_size
                end_y = start_y + cell_size

                # Check bounds
                if (end_x <= self.grid.shape[1] and end_y <= self.grid.shape[0]):
                    raw_cell = self.grid[start_y:end_y, start_x:end_x]
                    processed_cell = self.preprocess_cell(raw_cell)
                    raw_cells.append(raw_cell)
                    processed_cells.append(processed_cell)
                else:
                    raw_cells.append(None)
                    processed_cells.append(None)
        
        return raw_cells, processed_cells

    def process_grid(self, grid_image, debug=False):
        """
        Remove only gridlines that take up the entire width or height of the image.
        """
        if grid_image.dtype != np.uint8:
            if grid_image.dtype == np.float64:
                grid_image = (grid_image * 255).astype(np.uint8)
            else:
                grid_image = grid_image.astype(np.uint8)

        if len(grid_image.shape) == 3:
            gray = cv.cvtColor(grid_image, cv.COLOR_BGR2GRAY)
        else:
            gray = grid_image.copy()

        processed = gray.copy()
        h, w = gray.shape

        # First, enhance contrast to make lines more visible
        enhanced = cv.equalizeHist(gray)
        
        # Use adaptive thresholding to better detect lines
        binary = cv.adaptiveThreshold(enhanced, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
        
        # Detect horizontal lines using morphological operations
        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (w//8, 1))
        horizontal_lines = cv.morphologyEx(binary, cv.MORPH_OPEN, horizontal_kernel, iterations=1)
        
        # Detect vertical lines
        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, h//8))
        vertical_lines = cv.morphologyEx(binary, cv.MORPH_OPEN, vertical_kernel, iterations=1)
        
        # Create masks for detected lines
        _, horizontal_mask = cv.threshold(horizontal_lines, 50, 255, cv.THRESH_BINARY)
        _, vertical_mask = cv.threshold(vertical_lines, 50, 255, cv.THRESH_BINARY)
        
        # Dilate the masks to ensure we capture the full line width
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        horizontal_mask = cv.dilate(horizontal_mask, kernel, iterations=2)
        vertical_mask = cv.dilate(vertical_mask, kernel, iterations=2)
        
        # Combine masks
        grid_mask = cv.bitwise_or(horizontal_mask, vertical_mask)

        # Remove grid lines by inpainting
        result = cv.inpaint(processed, grid_mask, 3, cv.INPAINT_TELEA)

        if debug:
            self._display_grid_processing(
                gray,
                enhanced,
                binary,
                horizontal_mask,
                vertical_mask,
                grid_mask,
                result
            )

        return result

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
        black_percent_threshold = 0.15  # Lower threshold to be more aggressive in whitening lines
        edge_strip_width = 5            # Widen the strip to ensure full line coverage
        max_strips = 40                 # Allow more strips to be checked

        img = cell_image.copy()
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # img = (img.astype(np.float32) * 0.7).clip(0, 255).astype(np.uint8)
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

    def find_digit_in_cell(self, cell_image):
        """
        Finds the bounding rectangle of the largest contour (digit) in the cell image,
        crops to that region, and resizes back to the original cell_image size.
        """
        # Clean the image: binarize and dilate to connect digit parts
        # Convert to grayscale if needed
        if len(cell_image.shape) == 3:
            cell_gray = cv.cvtColor(cell_image, cv.COLOR_BGR2GRAY)
        else:
            cell_gray = cell_image
        _, cleaned = cv.threshold(cell_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        kernel = np.ones((5,5), np.uint8)
        dilation = cv.dilate(cleaned, kernel, iterations=1)
        # Find contours
        contours, _ = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(largest)
            # Crop to bounding box
            cropped = cell_image[y:y+h, x:x+w]
            # Resize back to original size
            resized = cv.resize(cropped, (cell_image.shape[1], cell_image.shape[0]), interpolation=cv.INTER_LINEAR)
            return resized
        else:
            # If no contour found, return the original image
            return cell_image

    def process_cells(self, cell_image, debug=False):
        """
        Process cell image to remove grid borders, clean up the image, and sharpen the digit
        """
        filtered_digit = self.find_digit_in_cell(cell_image)

        # Ensure the image is grayscale for adaptive thresholding
        if len(filtered_digit.shape) == 3:
            filtered_digit = cv.cvtColor(filtered_digit, cv.COLOR_BGR2GRAY)

        binary = cv.GaussianBlur(filtered_digit, (3, 3), 0)
        thresh = cv.adaptiveThreshold(binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                              cv.THRESH_BINARY, 11, 2)

        # Add a decent blur to the thresholded image to help smooth digit edges
        thresh = cv.GaussianBlur(thresh, (7, 7), 0)
        kernel = np.ones((3,3), np.uint8)
        binary = cv.dilate(thresh, kernel, iterations=1)
        thresh = cv.GaussianBlur(binary, (7, 7), 0)
        # # Simply sharpen digit
        # Invert colors
        binary = cv.bitwise_not(binary)
        # Erode the image
        kernel = np.ones((7,7), np.uint8)
        binary = cv.dilate(binary, kernel, iterations=1)
        kernel = np.ones((3,3), np.uint8)
        binary = cv.erode(binary, kernel, iterations=1)

        filtered_digit = binary

        # Resize to 28x28 for AI model compatibility
        filtered_digit = self._resize_cell(filtered_digit, target_size=(28, 28))

        return filtered_digit
        
    
    def _crop_cell(self, cell, crop_pct):
        height, width = cell.shape[:2]
        crop_x = int(width * crop_pct / 100)
        crop_y = int(height * crop_pct / 100)
        
        cropped = cell[crop_y:height-crop_y, crop_x:width-crop_x]
    
        return cropped

    def is_cell_blank(self, cell_image, threshold=None):
        """
        Optimized zero detection using edge/contour analysis.
        Returns True if the cell appears to be empty (digit 0), False if it contains a digit.
        
        Based on analysis of 1458 processed images showing:
        - Digit 0: 23.2 ± 33.4 edges, 0.5 ± 0.8 contours
        - Next lowest: 108.9 edges, 1.0 contours
        
        Args:
            cell_image: The cell image to analyze
            threshold: Not used (kept for compatibility)
        """
        # Convert to grayscale if needed
        if len(cell_image.shape) == 3 and cell_image.shape[2] == 3:
            gray = cv.cvtColor(cell_image, cv.COLOR_BGR2GRAY)
        else:
            gray = cell_image.copy()
        
        # Edge detection using Canny (same parameters as analysis)
        edges = cv.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        
        # Find contours
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Filter out very small contours (likely noise) - same as analysis
        min_contour_area = 5
        significant_contours = [c for c in contours if cv.contourArea(c) > min_contour_area]
        num_contours = len(significant_contours)
        
        # Optimal thresholds based on analysis and real-world testing:
        # Use a scoring system rather than strict AND logic
        edge_threshold = 87  # Conservative edge threshold
        contour_threshold = 1  # Conservative contour threshold
        
        # Scoring system: lower scores indicate higher likelihood of being zero
        edge_score = min(edge_pixels / edge_threshold, 2.0)  # Cap at 2.0
        contour_score = min(num_contours / contour_threshold, 2.0)  # Cap at 2.0
        
        # Combined score: lower is more likely to be zero
        combined_score = (edge_score + contour_score) / 2.0
        
        # Threshold for zero detection (tuned for balance)
        return combined_score < 1.0

    def analyze_cell_features(self, cell_image):
        """
        Analyze cell features for debugging and optimization.
        Returns detailed metrics about edge/contour characteristics.
        
        Args:
            cell_image: The cell image to analyze
            
        Returns:
            Dictionary with edge_pixels, num_contours, and zero_likelihood
        """
        # Convert to grayscale if needed
        if len(cell_image.shape) == 3 and cell_image.shape[2] == 3:
            gray = cv.cvtColor(cell_image, cv.COLOR_BGR2GRAY)
        else:
            gray = cell_image.copy()
        
        # Edge detection using Canny
        edges = cv.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        
        # Find contours
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Filter out very small contours
        min_contour_area = 5
        significant_contours = [c for c in contours if cv.contourArea(c) > min_contour_area]
        num_contours = len(significant_contours)
        
        # Calculate zero likelihood using scoring system
        edge_threshold = 87
        contour_threshold = 1
        
        edge_score = min(edge_pixels / edge_threshold, 2.0)
        contour_score = min(num_contours / contour_threshold, 2.0)
        combined_score = (edge_score + contour_score) / 2.0
        is_zero = combined_score < 1.0
        
        return {
            'edge_pixels': edge_pixels,
            'num_contours': num_contours,
            'edge_score': edge_score,
            'contour_score': contour_score,
            'combined_score': combined_score,
            'is_zero': is_zero,
            'edge_threshold': edge_threshold,
            'contour_threshold': contour_threshold
        }

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

    def _display_grid_processing(self, original, enhanced, binary, horizontal_mask, vertical_mask, grid_mask, result):
        """Display the grid processing steps for debugging"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original grid
        axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original Grid')
        axes[0, 0].axis('off')
        
        # Enhanced contrast
        axes[0, 1].imshow(enhanced, cmap='gray')
        axes[0, 1].set_title('Enhanced Contrast')
        axes[0, 1].axis('off')
        
        # Binary threshold
        axes[0, 2].imshow(binary, cmap='gray')
        axes[0, 2].set_title('Binary Threshold')
        axes[0, 2].axis('off')
        
        # Horizontal lines in green
        axes[1, 0].imshow(original, cmap='gray', alpha=0.7)
        axes[1, 0].imshow(horizontal_mask, cmap='Greens', alpha=0.8)
        axes[1, 0].set_title('Detected Horizontal Lines (Green)')
        axes[1, 0].axis('off')
        
        # Vertical lines in green
        axes[1, 1].imshow(original, cmap='gray', alpha=0.7)
        axes[1, 1].imshow(vertical_mask, cmap='Greens', alpha=0.8)
        axes[1, 1].set_title('Detected Vertical Lines (Green)')
        axes[1, 1].axis('off')
        
        # Final result
        axes[1, 2].imshow(result, cmap='gray')
        axes[1, 2].set_title('Processed Grid (No Lines)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()

    def display_processed_grid(self):
        """Display the processed grid image"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(self.original_grid, cmap='gray')
        axes[0].set_title('Original Grid')
        axes[0].axis('off')
        
        axes[1].imshow(self.grid, cmap='gray')
        axes[1].set_title('Processed Grid (Gridlines Removed)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

    def get_processed_grid(self):
        """Return the processed grid image"""
        return self.grid.copy()

    def get_original_grid(self):
        """Return the original grid image"""
        return self.original_grid.copy()

