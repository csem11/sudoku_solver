import numpy as np
from PIL import Image
import os
import keras
import sys
import cv2 as cv
from pathlib import Path

# Add src to path to import naming utilities
sys.path.append(str(Path(__file__).parent.parent))
from utils.naming import parse_filename


def retrieve_digit_dataset(data_dir, return_categorical=True, manual_proportion=0.25, include_zero=True):
    """
    Load digit dataset from directory, including both manual and synthetic digit images.

    Args:
        data_dir: Path to directory containing digit images (should contain 'manual' and/or 'synthetic' subdirs)
        return_categorical: If True, convert labels to categorical format
        manual_proportion: Proportion of data that should be manual (between 0 and 1)
        include_zero: If False, exclude digit 0 from the dataset (default: False)

    Returns:
        X: Image array (samples, height, width, channels)
        y: Label array (categorical if return_categorical=True, else string labels)
    """
    manual_images = []
    manual_labels = []
    synthetic_images = []
    synthetic_labels = []

    # Collect manual images from processed directory
    manual_dir = os.path.join(data_dir, 'manual', 'processed')
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
    else:
        print(f"Warning: Manual processed directory not found at {manual_dir}")

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

    # Filter out digit 0 if include_zero is False and remap labels
    if not include_zero:
        filtered_images = []
        filtered_labels = []
        for img, label in zip(images, labels):
            if int(label) != 0:  # Exclude digit 0
                filtered_images.append(img)
                # Remap labels: 1->0, 2->1, 3->2, ..., 9->8
                remapped_label = int(label) - 1
                filtered_labels.append(remapped_label)
        images = filtered_images
        labels = filtered_labels
        
        if len(images) == 0:
            print("Warning: No non-zero digits found in dataset")
            return np.array([]), np.array([])

    # Ensure all images are 28x28 and add debugging
    print(f"Processing {len(images)} images...")
    processed_images = []
    for i, img in enumerate(images):
        if img is None:
            print(f"Warning: Image {i} is None, skipping...")
            continue
            
        # Ensure image is 2D
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        
        # Resize to 28x28 if not already
        if img.shape != (28, 28):
            img = cv.resize(img, (28, 28), interpolation=cv.INTER_AREA)
        
        processed_images.append(img)
    
    # Update labels to match processed images
    labels = labels[:len(processed_images)]
    
    if len(processed_images) == 0:
        return np.array([]), np.array([])

    indices = np.random.permutation(len(processed_images))
    images = [processed_images[i] for i in indices]
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
        # Adjust number of classes based on include_zero parameter
        num_classes = 9 if not include_zero else 10
        y = keras.utils.to_categorical(y_int, num_classes=num_classes)

    return X, y

