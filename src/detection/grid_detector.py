import cv2 as cv
import numpy as np



class GridDetector:
    def __init__(self):
        pass

    def detect(self, frame):
        """Detect grid contour in the given frame"""
        contours = self._find_contours(frame)
        if not contours:
            return None

        return contours[0]

    def find_grid(self, frame):
        """Legacy method - use detect() instead"""
        return self.detect(frame)
    
    def highlight_grid(self, frame, contour, color=(0, 255, 0), thickness=2):
        """Highlight the detected grid contour on the frame"""
        if contour is not None:
            # Draw the contour
            cv.drawContours(frame, [contour], -1, color, thickness)
            
            # Draw corner points
            corners = contour.reshape(4, 2)
            for i, corner in enumerate(corners):
                cv.circle(frame, tuple(corner.astype(int)), 5, color, -1)
                cv.putText(frame, str(i), tuple(corner.astype(int) + 10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame

    def _find_contours(self, frame, min_area=10000, max_area=800000):
        contours, _ = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        size_filtered = []
        for contour in contours:
            area = cv.contourArea(contour)
            if min_area < area < max_area:
                size_filtered.append(contour)
        
        grid_canidates = self._rank_grid_canidates(size_filtered)

        return grid_canidates


    def _rank_grid_canidates(self, contours, min_aspect_ratio=0.8, max_aspect_ratio=1.2):
        ranked_squares = []

        for contour in contours:
            perimeter = cv.arcLength(contour, True)
            epsilon = 0.02 * perimeter
            approx = cv.approxPolyDP(contour, epsilon, True)
            
            # Must be roughly 4-sided (quadrilateral)
            if len(approx) == 4:
                # Check if roughly square (aspect ratio)
                x, y, w, h = cv.boundingRect(contour)
                aspect_ratio = float(w / h)
                corners = approx.reshape(4, 2)
                score = abs(1.0 - aspect_ratio)
                ranked_squares.append((score, corners))

        ranked_squares.sort(key=lambda x: x[0])
        return [contour for score, contour in ranked_squares]


 