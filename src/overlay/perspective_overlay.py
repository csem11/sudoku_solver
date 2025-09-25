"""
Perspective-Aware Overlay System

Based on the Mirda81/Sudoku repository approach, this module provides
proper inverse perspective transformation for accurate digit positioning.
"""

import cv2 as cv
import numpy as np
from typing import Tuple, Optional, Dict, Any


class PerspectiveOverlay:
    """
    Handles overlay positioning using proper inverse perspective transformation.
    Inspired by the Mirda81/Sudoku repository approach.
    """
    
    def __init__(self, output_size: int = 600):
        self.output_size = output_size
        
        # Define target square corners for transformation
        self.target_corners = np.array([
            [0, 0],                           # top-left
            [output_size, 0],                 # top-right
            [output_size, output_size],       # bottom-right
            [0, output_size]                  # bottom-left
        ], dtype=np.float32)
    
    def get_cell_positions(self, frame: np.ndarray, grid_corners: np.ndarray) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Get precise cell positions using inverse perspective transformation.
        
        This approach:
        1. Creates a perfect square grid in transformed space
        2. Calculates cell positions in that space
        3. Uses inverse transformation to map back to original coordinates
        
        Args:
            frame: Original video frame
            grid_corners: Detected grid corners (4x2 array)
            
        Returns:
            Dictionary mapping (row, col) to cell position info
        """
        # Order corners consistently
        ordered_corners = self._order_corners(grid_corners)
        
        # Calculate transformation matrix from original to target
        transform_matrix = cv.getPerspectiveTransform(ordered_corners, self.target_corners)
        
        # Calculate inverse transformation matrix
        inverse_matrix = cv.getPerspectiveTransform(self.target_corners, ordered_corners)
        
        # Calculate cell size in transformed space
        cell_size = self.output_size // 9
        
        cell_positions = {}
        
        for row in range(9):
            for col in range(9):
                # Calculate cell boundaries in transformed space
                start_x = col * cell_size
                end_x = start_x + cell_size
                start_y = row * cell_size
                end_y = start_y + cell_size
                
                # Get the four corners of this cell in transformed space
                cell_corners = np.array([
                    [start_x, start_y],      # top-left
                    [end_x, start_y],        # top-right
                    [end_x, end_y],          # bottom-right
                    [start_x, end_y]         # bottom-left
                ], dtype=np.float32)
                
                # Transform cell corners back to original coordinates
                original_corners = cv.perspectiveTransform(
                    cell_corners.reshape(-1, 1, 2), inverse_matrix
                ).reshape(-1, 2)
                
                # Calculate cell center and boundaries
                center_x = np.mean(original_corners[:, 0])
                center_y = np.mean(original_corners[:, 1])
                
                left = np.min(original_corners[:, 0])
                right = np.max(original_corners[:, 0])
                top = np.min(original_corners[:, 1])
                bottom = np.max(original_corners[:, 1])
                
                cell_positions[(row, col)] = {
                    'center': (int(center_x), int(center_y)),
                    'left': int(left),
                    'right': int(right),
                    'top': int(top),
                    'bottom': int(bottom),
                    'width': right - left,
                    'height': bottom - top,
                    'corners': original_corners
                }
        
        return cell_positions
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners consistently: top-left, top-right, bottom-right, bottom-left
        Based on the Mirda81/Sudoku approach.
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
    
    def calculate_grid_rotation(self, grid_corners: np.ndarray) -> float:
        """Calculate the rotation angle of the grid."""
        ordered_corners = self._order_corners(grid_corners)
        
        # Get top edge vector (from top-left to top-right)
        top_left = ordered_corners[0]
        top_right = ordered_corners[1]
        top_edge_vector = top_right - top_left
        
        # Calculate angle of top edge
        angle_rad = np.arctan2(top_edge_vector[1], top_edge_vector[0])
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def draw_debug_overlay(self, frame: np.ndarray, grid_corners: np.ndarray, 
                          cell_positions: Dict[Tuple[int, int], Dict[str, Any]]) -> np.ndarray:
        """Draw debug overlay showing grid structure and cell positions."""
        debug_frame = frame.copy()
        
        # Draw grid corners
        corners = grid_corners.reshape(4, 2)
        for i, corner in enumerate(corners):
            cv.circle(debug_frame, tuple(corner.astype(int)), 6, (0, 255, 255), -1)
            cv.putText(debug_frame, str(i), tuple(corner.astype(int) + 8), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw grid outline
        cv.polylines(debug_frame, [grid_corners.reshape(-1, 1, 2).astype(int)], True, (0, 255, 255), 2)
        
        # Draw cell boundaries for first few cells
        for (row, col), cell_info in cell_positions.items():
            if row < 3 and col < 3:  # Only show first 9 cells
                # Draw cell rectangle
                cv.rectangle(debug_frame, 
                           (cell_info['left'], cell_info['top']),
                           (cell_info['right'], cell_info['bottom']), 
                           (255, 0, 0), 1)
                
                # Draw cell center
                cv.circle(debug_frame, cell_info['center'], 3, (0, 0, 255), -1)
                
                # Draw cell number
                cv.putText(debug_frame, f"{row},{col}", 
                          (cell_info['center'][0] + 5, cell_info['center'][1] - 5), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return debug_frame
    
    def draw_rotated_text(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                         font: int, font_scale: float, color: Tuple[int, int, int], 
                         thickness: int, angle: float) -> None:
        """Draw rotated text on the frame using proper transformation."""
        x, y = position
        
        # Get text size
        (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, thickness)
        
        # Create rotation matrix centered at the text position
        rotation_matrix = cv.getRotationMatrix2D((x, y), angle, 1.0)
        
        # Create text image
        text_img = np.zeros((text_height + baseline, text_width, 3), dtype=np.uint8)
        cv.putText(text_img, text, (0, text_height), font, font_scale, color, thickness)
        
        # Rotate the text image
        rotated_text = cv.warpAffine(text_img, rotation_matrix, 
                                   (text_width, text_height + baseline))
        
        # Calculate position to center the rotated text
        h, w = rotated_text.shape[:2]
        x_offset = x - w // 2
        y_offset = y - h // 2
        
        # Ensure coordinates are within frame bounds
        x_offset = max(0, min(x_offset, frame.shape[1] - w))
        y_offset = max(0, min(y_offset, frame.shape[0] - h))
        
        # Overlay the rotated text
        mask = rotated_text > 0
        frame[y_offset:y_offset+h, x_offset:x_offset+w][mask] = rotated_text[mask]
