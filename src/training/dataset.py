import numpy as np
from PIL import Image
import os
import keras


def retrieve_digit_dataset(data_dir, return_categorical=True):
    """
    Load digit dataset from directory.
    
    Args:
        data_dir: Path to directory containing digit images
        return_categorical: If True, convert labels to categorical format
    
    Returns:
        X: Image array (samples, height, width, channels)
        y: Label array (categorical if return_categorical=True, else string labels)
    """
    images = []
    labels = []

    for img_file in os.listdir(data_dir):
        if img_file.endswith('.png'):
            img = Image.open(os.path.join(data_dir, img_file))
            digit = img_file[6]

            images.append(img)
            labels.append(digit)

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

