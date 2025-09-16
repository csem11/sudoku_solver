import numpy as np
from PIL import Image
import os
import keras
import sys
from pathlib import Path

# Add src to path to import naming utilities
sys.path.append(str(Path(__file__).parent.parent))
from utils.naming import parse_filename


def retrieve_digit_dataset(data_dir, return_categorical=True):
    """
    Load digit dataset from directory, including both manual and synthetic digit images.

    Args:
        data_dir: Path to directory containing digit images (should contain 'manual' and/or 'synthetic' subdirs)
        return_categorical: If True, convert labels to categorical format

    Returns:
        X: Image array (samples, height, width, channels)
        y: Label array (categorical if return_categorical=True, else string labels)
    """
    images = []
    labels = []

    # Only look in the 'training' subdir for image files
    subdir_path = os.path.join(data_dir, 'training')
    if os.path.exists(subdir_path):
        for img_file in os.listdir(subdir_path):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                metadata = parse_filename(img_file)
                if metadata is None:
                    print(f"Warning: Could not parse filename {img_file}, skipping...")
                    continue

                img = Image.open(os.path.join(subdir_path, img_file)).convert('L')
                images.append(np.array(img))
                labels.append(metadata['digit'])

    X = np.array(images)
    y = np.array(labels)
    
    # Convert to grayscale if RGB
    if X.shape[-1] == 3:
        X = np.mean(X, axis=-1, keepdims=True)
    elif len(X.shape) == 3:
        # If already grayscale but missing channel dimension, add it
        X = np.expand_dims(X, axis=-1)
    
    # Normalize pixel values to [0, 1]
    X = X.astype(np.float32) / 255.0
    
    # Convert labels to categorical if requested
    if return_categorical:
        # Convert string labels to integers first
        y_int = y.astype(int)
        y = keras.utils.to_categorical(y_int, num_classes=10)
    
    return X, y

