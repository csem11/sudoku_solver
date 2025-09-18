import numpy as np
from PIL import Image
import os
import keras
import sys
from pathlib import Path

# Add src to path to import naming utilities
sys.path.append(str(Path(__file__).parent.parent))
from utils.naming import parse_filename


def retrieve_digit_dataset(data_dir, return_categorical=True, manual_proportion=0.25):
    """
    Load digit dataset from directory, including both manual and synthetic digit images.

    Args:
        data_dir: Path to directory containing digit images (should contain 'manual' and/or 'synthetic' subdirs)
        return_categorical: If True, convert labels to categorical format
        manual_proportion: Proportion of data that should be manual (between 0 and 1)

    Returns:
        X: Image array (samples, height, width, channels)
        y: Label array (categorical if return_categorical=True, else string labels)
    """
    manual_images = []
    manual_labels = []
    synthetic_images = []
    synthetic_labels = []

    # Collect manual images
    manual_dir = os.path.join(data_dir, 'manual')
    if os.path.exists(manual_dir):
        for img_file in os.listdir(manual_dir):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                metadata = parse_filename(img_file)
                if metadata is None:
                    print(f"Warning: Could not parse filename {img_file}, skipping...")
                    continue
                img = Image.open(os.path.join(manual_dir, img_file)).convert('L')
                manual_images.append(np.array(img))
                manual_labels.append(metadata['digit'])

    # Collect synthetic images
    synthetic_dir = os.path.join(data_dir, 'synthetic')
    if os.path.exists(synthetic_dir):
        for img_file in os.listdir(synthetic_dir):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                metadata = parse_filename(img_file)
                if metadata is None:
                    print(f"Warning: Could not parse filename {img_file}, skipping...")
                    continue
                img = Image.open(os.path.join(synthetic_dir, img_file)).convert('L')
                synthetic_images.append(np.array(img))
                synthetic_labels.append(metadata['digit'])

    # Determine number of manual and synthetic samples to use
    total_manual = len(manual_images)
    total_synthetic = len(synthetic_images)
    total = total_manual + total_synthetic

    if total == 0:
        return np.array([]), np.array([])

    # Calculate how many manual samples to use
    n_manual = int(round(manual_proportion * total))
    n_manual = min(n_manual, total_manual)
    n_synthetic = total - n_manual
    n_synthetic = min(n_synthetic, total_synthetic)

    # Shuffle before sampling
    if total_manual > 0:
        manual_indices = np.random.permutation(total_manual)[:n_manual]
        manual_images = [manual_images[i] for i in manual_indices]
        manual_labels = [manual_labels[i] for i in manual_indices]
    else:
        manual_images = []
        manual_labels = []

    if total_synthetic > 0:
        synthetic_indices = np.random.permutation(total_synthetic)[:n_synthetic]
        synthetic_images = [synthetic_images[i] for i in synthetic_indices]
        synthetic_labels = [synthetic_labels[i] for i in synthetic_indices]
    else:
        synthetic_images = []
        synthetic_labels = []

    # Combine and shuffle
    images = manual_images + synthetic_images
    labels = manual_labels + synthetic_labels

    if len(images) == 0:
        return np.array([]), np.array([])

    indices = np.random.permutation(len(images))
    images = [images[i] for i in indices]
    labels = [labels[i] for i in indices]

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
        y_int = y.astype(int)
        y = keras.utils.to_categorical(y_int, num_classes=10)

    return X, y

