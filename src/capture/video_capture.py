import cv2 as cv 
import numpy as np
from typing import Optional, Tuple

class VideoCapture:
    def __init__(self, source: int = 0, width: int = 640, height: int = 480):
        """
        Initialize video capture
        
        Args:
            source: Camera index (0 for default camera) or video file path
            width: Frame width
            height: Frame height
        """
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.is_running = False
    
    def start(self) -> bool:
        """Start video capture"""
        self.cap = cv.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            return False
        
        # Set resolution
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
        
        self.is_running = True
        return True
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_running or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        
        return True, frame
    
    def stop(self):
        """Stop video capture and release resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
    
    def get_frame_size(self) -> Tuple[int, int]:
        """Get current frame dimensions"""
        if self.cap:
            width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return self.width, self.height
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

# Usage example
if __name__ == "__main__":
    with VideoCapture() as video:
        while True:
            success, frame = video.read_frame()
            if success:
                cv.imshow('Video Feed', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
    
    cv.destroyAllWindows()