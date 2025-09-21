import numpy as np
import tensorflow as tf

from pathlib import Path
import sys

# Add the src directory to the path so we can import from training module
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from training.model import DigitClassifier

class GridGenerator:
    def __init__(self):
        model_path = Path(__file__).parent.parent.parent / "models" / "digit_classifier.keras"

        self.model = tf.keras.models.load_model(model_path)
        
        # Prediction statistics
        self.pred_stats = {
            'highest_confidence': 0.0,
            'lowest_confidence': 1.0,
            'average_confidence': 0.0,
            'high_confidence_count': 0,  # predictions >= 0.8
            'low_confidence_count': 0,   # predictions < 0.5
            'total_predictions': 0
        }
    
    def generate_grid(self, cell_images, include_prob_grid=True):
        # Convert RGB images to grayscale if needed
        if len(cell_images.shape) == 4 and cell_images.shape[-1] == 3:
            # Convert RGB to grayscale
            cell_images = np.mean(cell_images, axis=-1, keepdims=True)
        
        # Ensure images are in the correct format (28x28 grayscale)
        if cell_images.shape[1:3] != (28, 28):
            import cv2 as cv
            resized_images = []
            for img in cell_images:
                if len(img.shape) == 3 and img.shape[-1] == 1:
                    img = img.squeeze()
                resized = cv.resize(img, (28, 28))
                resized_images.append(resized)
            cell_images = np.array(resized_images)
            if len(cell_images.shape) == 3:
                cell_images = cell_images.reshape(-1, 28, 28, 1)
        
        # Convert to float32 and normalize to [0, 1]
        cell_images = cell_images.astype(np.float32) / 255.0
        
        predictions = self.model.predict(cell_images)
        
        # Update prediction statistics
        self._update_prediction_stats(predictions)

        predicted_digits = np.argmax(predictions, axis=1)
        grid = predicted_digits.reshape(9, 9)

        if include_prob_grid:
            digit_probability = np.max(predictions, axis=1)
            prob_grid = digit_probability.reshape(9, 9)

            return grid, prob_grid
        
        return grid

    def _update_prediction_stats(self, predictions):
        """Update prediction statistics based on the confidence values"""
        if predictions is None or len(predictions) == 0:
            return
            
        # Get the maximum confidence for each prediction
        max_confidences = np.max(predictions, axis=1)
        
        # Update statistics
        self.pred_stats['highest_confidence'] = float(np.max(max_confidences))
        self.pred_stats['lowest_confidence'] = float(np.min(max_confidences))
        self.pred_stats['average_confidence'] = float(np.mean(max_confidences))
        self.pred_stats['high_confidence_count'] = int(np.sum(max_confidences >= 0.8))
        self.pred_stats['low_confidence_count'] = int(np.sum(max_confidences < 0.5))
        self.pred_stats['total_predictions'] = len(max_confidences)

    def get_prediction_stats(self):
        """Return current prediction statistics"""
        return self.pred_stats.copy()

    def print_prediction_stats(self):
        """Print prediction statistics in a formatted way"""
        stats = self.pred_stats
        print("\n=== Prediction Statistics ===")
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Highest confidence: {stats['highest_confidence']:.3f}")
        print(f"Lowest confidence: {stats['lowest_confidence']:.3f}")
        print(f"Average confidence: {stats['average_confidence']:.3f}")
        print(f"High confidence (≥0.8): {stats['high_confidence_count']}")
        print(f"Low confidence (<0.5): {stats['low_confidence_count']}")
        print("=" * 30)

    def show_board(self, grid, prob_grid=None, cell_size=50):
        """
        Visualize the Sudoku grid with predicted digits and color squares based on confidence.
        Green = high confidence, Red = low confidence.
        Includes prediction statistics on the right side.
        """
        import cv2 as cv
        
        # Calculate board dimensions with space for stats
        stats_width = 200
        board_img = np.ones((9 * cell_size, 9 * cell_size + stats_width, 3), dtype=np.uint8) * 255

        for i in range(9):
            for j in range(9):
                x1, y1 = j * cell_size, i * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size

                # Determine color based on probability
                if prob_grid is not None:
                    prob = prob_grid[i, j]
                    # Interpolate between red (low) and green (high)
                    green = int(255 * prob)
                    red = int(255 * (1 - prob))
                    color = (0, green, red)
                else:
                    color = (200, 200, 200)

                cv.rectangle(board_img, (x1, y1), (x2, y2), color, -1)

                # Draw digit if not zero
                digit = int(grid[i, j])
                if digit != 0:
                    text = str(digit)
                    font = cv.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.2
                    thickness = 2
                    text_size, _ = cv.getTextSize(text, font, font_scale, thickness)
                    text_x = x1 + (cell_size - text_size[0]) // 2
                    text_y = y1 + (cell_size + text_size[1]) // 2
                    cv.putText(board_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv.LINE_AA)

        # Draw grid lines
        for i in range(10):
            thickness = 2 if i % 3 == 0 else 1
            cv.line(board_img, (0, i * cell_size), (9 * cell_size, i * cell_size), (0, 0, 0), thickness)
            cv.line(board_img, (i * cell_size, 0), (i * cell_size, 9 * cell_size), (0, 0, 0), thickness)

        # Add prediction statistics on the right side
        self._draw_prediction_stats(board_img, 9 * cell_size + 10, cell_size)

        cv.imshow("Predicted Sudoku Board", board_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def _draw_prediction_stats(self, img, start_x, cell_size):
        """Draw prediction statistics on the right side of the board image"""
        import cv2 as cv
        
        stats = self.pred_stats
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 25
        y_start = 20
        
        # Title
        cv.putText(img, "Prediction Stats:", (start_x, y_start), font, font_scale + 0.1, (0, 0, 0), thickness + 1)
        y_start += line_height + 10
        
        # Statistics
        stats_text = [
            f"Total: {stats['total_predictions']}",
            f"High conf: {stats['high_confidence_count']}",
            f"Low conf: {stats['low_confidence_count']}",
            f"Max: {stats['highest_confidence']:.3f}",
            f"Min: {stats['lowest_confidence']:.3f}",
            f"Avg: {stats['average_confidence']:.3f}"
        ]
        
        for text in stats_text:
            cv.putText(img, text, (start_x, y_start), font, font_scale, (50, 50, 50), thickness)
            y_start += line_height
