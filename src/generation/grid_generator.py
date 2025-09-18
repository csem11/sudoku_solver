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

        predicted_digits = np.argmax(predictions, axis=1)
        grid = predicted_digits.reshape(9, 9)

        if include_prob_grid:
            digit_probability = np.max(predictions, axis=1)
            prob_grid = digit_probability.reshape(9, 9)

            return grid, prob_grid
        
        return grid

    def show_board(self, grid, prob_grid=None, cell_size=50):
        """
        Visualize the Sudoku grid with predicted digits and color squares based on confidence.
        Green = high confidence, Red = low confidence.
        """
        import cv2 as cv
        board_img = np.ones((9 * cell_size, 9 * cell_size, 3), dtype=np.uint8) * 255

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

        cv.imshow("Predicted Sudoku Board", board_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
