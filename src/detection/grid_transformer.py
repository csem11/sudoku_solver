import cv2 as cv
import numpy as np

class GridTransformer:
    def __init__(self, output_size: int = 600):

        self.output_size = output_size
    
    def transform(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        # Order corners consistently
        ordered_corners = self._order_corners(corners)
        
        # Define target square corners
        target_corners = np.array([
            [0, 0],                           # top-left
            [self.output_size, 0],            # top-right
            [self.output_size, self.output_size], # bottom-right
            [0, self.output_size]             # bottom-left
        ], dtype=np.float32)
        
        # Calculate transformation matrix
        matrix = cv.getPerspectiveTransform(ordered_corners, target_corners)
        
        # Apply transformation
        warped = cv.warpPerspective(frame, matrix, (self.output_size, self.output_size))
        
        return warped
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners consistently: top-left, top-right, bottom-right, bottom-left
        """
        corners = corners.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum of coordinates - top-left smallest, bottom-right largest
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]  # top-left
        rect[2] = corners[np.argmax(s)]  # bottom-right
        
        # Difference of coordinates - top-right smallest diff, bottom-left largest
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]  # top-right
        rect[3] = corners[np.argmax(diff)]  # bottom-left
        
        return rect
