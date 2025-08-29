import cv2 as cv
import numpy as np



class GridDetector:
    def __init__(self, frame):
        self.frame = frame

    def find_grid(self):
        contours = self._find_contours()
        if not contours:
            return []

        return contours[0]

    def _find_contours(self, min_area=10000, max_area=800000):

        contours, _ = cv.findContours(self.frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

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


 