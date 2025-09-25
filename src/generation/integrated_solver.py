import cv2 as cv
import numpy as np
from typing import Optional, Tuple, List
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from .grid_generator import GridGenerator
from .sudoku import Sudoku
from .board_visualizer import SudokuBoardVisualizer
from detection.grid_detector import GridDetector
from detection.cell_extactor import CellExtractor
from detection.grid_transformer import GridTransformer

class IntegratedSudokuSolver:
    """
    Integrated Sudoku solver that detects grids from images, extracts digits,
    solves the puzzle, and displays the solution directly on the original image.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the integrated solver with all required components.
        
        Args:
            model_path: Path to the digit classification model
        """
        self.grid_generator = GridGenerator(model_path)
        self.board_visualizer = SudokuBoardVisualizer()
        self.grid_detector = GridDetector()
        self.grid_transformer = GridTransformer()
        
        print("Integrated Sudoku Solver initialized successfully")
    
    def solve_from_image(self, 
                        image_path: str,
                        show_comparison: bool = True,
                        save_result: bool = False,
                        output_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve a Sudoku puzzle from an image file.
        
        Args:
            image_path: Path to the image containing the Sudoku puzzle
            show_comparison: Whether to show side-by-side comparison
            save_result: Whether to save the result image
            output_path: Path to save the result (if save_result is True)
            
        Returns:
            Tuple of (original_image, original_puzzle, solved_puzzle)
        """
        # Load image
        original_image = cv.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        print(f"Processing image: {image_path}")
        
        # Detect and solve the puzzle
        original_puzzle, solved_puzzle, grid_corners = self._detect_and_solve(original_image)
        
        # Display results
        if show_comparison and original_puzzle is not None and solved_puzzle is not None:
            self.board_visualizer.visualize_solution_comparison(
                original_puzzle, solved_puzzle, f"Sudoku Solution - {Path(image_path).name}"
            )
        
        # Create overlay image with solved digits (only if puzzle was solved)
        if original_puzzle is not None and solved_puzzle is not None:
            result_image = self._create_overlay_image(
                original_image, original_puzzle, solved_puzzle, grid_corners
            )
        else:
            result_image = original_image
        
        # Save result if requested
        if save_result:
            if output_path is None:
                output_path = f"solved_{Path(image_path).name}"
            cv.imwrite(output_path, result_image)
            print(f"Result saved to: {output_path}")
        
        return original_image, original_puzzle, solved_puzzle
    
    def solve_from_camera_frame(self, 
                               frame: np.ndarray,
                               show_comparison: bool = False) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Solve a Sudoku puzzle from a camera frame.
        
        Args:
            frame: Camera frame containing the Sudoku puzzle
            show_comparison: Whether to show side-by-side comparison
            
        Returns:
            Tuple of (result_image, original_puzzle, solved_puzzle) or (None, puzzle, solution) if no grid detected
        """
        print("Processing camera frame...")
        
        # Detect and solve the puzzle
        original_puzzle, solved_puzzle, grid_corners = self._detect_and_solve(frame)
        
        if original_puzzle is None:
            print("No Sudoku grid detected in frame")
            return None, np.array([]), np.array([])
        
        # Display results
        if show_comparison:
            self.board_visualizer.visualize_solution_comparison(
                original_puzzle, solved_puzzle, "Sudoku Solution - Camera"
            )
        
        # Create overlay image with solved digits
        result_image = self._create_overlay_image(
            frame, original_puzzle, solved_puzzle, grid_corners
        )
        
        return result_image, original_puzzle, solved_puzzle
    
    def _detect_and_solve(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect grid, extract digits, and solve the Sudoku puzzle.
        
        Returns:
            Tuple of (original_puzzle, solved_puzzle, grid_corners)
        """
        # Preprocess image for grid detection
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blurred, 50, 150)
        
        # Detect grid
        grid_contour = self.grid_detector.detect(edges)
        if grid_contour is None:
            print("No grid detected")
            return None, None, None
        
        print("Grid detected, extracting cells...")
        
        # Extract grid corners for perspective transformation
        grid_corners = self._extract_grid_corners(grid_contour)
        
        # Transform the grid to a square
        transformed_grid = self.grid_transformer.transform_grid(image, grid_corners)
        if transformed_grid is None:
            print("Failed to transform grid")
            return None, None, None
        
        # Extract cells from the transformed grid
        cell_extractor = CellExtractor(transformed_grid)
        cell_images = cell_extractor.processed_cells
        
        if len(cell_images) != 81:
            print(f"Expected 81 cells, got {len(cell_images)}")
            return None, None, None
        
        print("Extracting digits from cells...")
        
        # Generate puzzle from cell images
        original_puzzle, prob_grid = self.grid_generator.generate_grid(cell_images, include_prob_grid=True)
        
        # Print prediction statistics
        self.grid_generator.print_prediction_stats()
        
        # Check if puzzle is valid
        if np.all(original_puzzle == 0):
            print("No digits detected in puzzle")
            return None, None, None
        
        print("Original puzzle detected:")
        self._print_puzzle(original_puzzle)
        
        # Solve the puzzle
        print("Solving puzzle...")
        solved_puzzle = self._solve_puzzle(original_puzzle)
        
        if solved_puzzle is None:
            print("Puzzle could not be solved")
            return original_puzzle, None, grid_corners
        
        print("Puzzle solved successfully!")
        print("Solution:")
        self._print_puzzle(solved_puzzle)
        
        return original_puzzle, solved_puzzle, grid_corners
    
    def _extract_grid_corners(self, contour: np.ndarray) -> np.ndarray:
        """
        Extract the four corners of the grid contour.
        """
        # Approximate the contour to get the corners
        epsilon = 0.02 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        
        if len(approx) != 4:
            # If we don't have exactly 4 corners, try a different approach
            # Find the convex hull and get the extreme points
            hull = cv.convexHull(contour)
            if len(hull) >= 4:
                # Find the 4 extreme points (top-left, top-right, bottom-right, bottom-left)
                hull = hull.reshape(-1, 2)
                corners = self._find_four_corners(hull)
            else:
                raise ValueError("Could not find 4 corners")
        else:
            corners = approx.reshape(4, 2)
        
        return corners
    
    def _find_four_corners(self, points: np.ndarray) -> np.ndarray:
        """
        Find the 4 corners from a set of points.
        """
        # Calculate the centroid
        centroid = np.mean(points, axis=0)
        
        # Sort points by angle from centroid
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        
        # Take every 4th point to get roughly 4 corners
        if len(sorted_points) >= 4:
            step = len(sorted_points) // 4
            corners = np.array([
                sorted_points[0],
                sorted_points[step],
                sorted_points[step * 2],
                sorted_points[step * 3]
            ])
        else:
            corners = sorted_points
        
        return corners
    
    def _solve_puzzle(self, puzzle: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve the Sudoku puzzle using the Sudoku class.
        """
        try:
            # Create a Sudoku instance with the detected puzzle
            sudoku_solver = Sudoku(board=puzzle.copy())
            
            # Solve the puzzle
            sudoku_solver.solve_board()
            
            return sudoku_solver.board
            
        except Exception as e:
            print(f"Error solving puzzle: {e}")
            return None
    
    def _create_overlay_image(self,
                            original_image: np.ndarray,
                            original_puzzle: np.ndarray,
                            solved_puzzle: np.ndarray,
                            grid_corners: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create an image with solved digits overlaid on the original image.
        """
        return self.board_visualizer.overlay_solved_puzzle(
            original_image, None, original_puzzle, solved_puzzle, grid_corners
        )
    
    def _print_puzzle(self, puzzle: np.ndarray) -> None:
        """
        Print the puzzle in a readable format.
        """
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("- - - + - - - + - - -")
            row_str = ""
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    row_str += "| "
                if puzzle[i, j] == 0:
                    row_str += ". "
                else:
                    row_str += f"{puzzle[i, j]} "
            print(row_str.rstrip())
        print()
    
    def process_video_stream(self, camera_index: int = 0) -> None:
        """
        Process a video stream from a camera and solve Sudoku puzzles in real-time.
        """
        cap = cv.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        print("Press 's' to solve current frame, 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Display the current frame
            cv.imshow("Sudoku Solver - Press 's' to solve, 'q' to quit", frame)
            
            key = cv.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Solve the current frame
                result_image, original_puzzle, solved_puzzle = self.solve_from_camera_frame(frame)
                
                if result_image is not None:
                    cv.imshow("Solved Sudoku", result_image)
                    print("Press any key to continue...")
                    cv.waitKey(0)
        
        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    solver = IntegratedSudokuSolver()
    
    # Process an image file
    # solver.solve_from_image("path/to/sudoku_image.jpg", show_comparison=True, save_result=True)
    
    # Or process video stream
    # solver.process_video_stream()
