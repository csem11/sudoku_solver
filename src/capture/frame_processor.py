import cv2 as cv
import numpy as np
from typing import Tuple, Optional


class FrameProcessor:
    def __init__(self, blur_kernel: int = 5, threshold_type: str = 'adaptive'):
        """
        Initialize frame processor
        
        Args:
            blur_kernel: Size for Gaussian blur (odd number)
            threshold_type: 'adaptive' or 'otsu'
        """
        self.blur_kernel = blur_kernel
        self.threshold_type = threshold_type
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to frame
        
        Args:
            frame: Input BGR frame or grayscale frame
            
        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply thresholding
        binary = cv.adaptiveThreshold(gray, 255, 
                                cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv.THRESH_BINARY_INV, 15, 5)
        
        return binary

