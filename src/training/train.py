#!/usr/bin/env python3
"""
Training script for digit classification model.
"""

import numpy as np
import keras
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from training.dataset import retrieve_digit_dataset
from training.model import DigitClassifier




def create_model(input_shape=(28, 28, 1), num_classes=10):
    """Create and compile the digit classification model."""
    model = DigitClassifier(input_shape=input_shape, num_classes=num_classes)
    
    # Build the model by calling it with sample input
    sample_input = keras.Input(shape=input_shape)
    _ = model(sample_input)
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def train_model(data_dir=None, 
                batch_size=32, 
                epochs=100, 
                validation_split=0.2,
                save_model=True,
                model_path='models/digit_classifier.keras'):
    """Train the digit classification model."""
    
    # Set default data directory if not provided
    if data_dir is None:
        # Get the project root directory (2 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / 'data' / 'digits'
        data_dir = str(data_dir)
    
    print(f"Loading dataset from: {data_dir}")
    X, y = retrieve_digit_dataset(data_dir, return_categorical=True)
    print(f"Loaded {len(X)} samples")
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=validation_split, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Create model
    print("Creating model...")
    input_shape = X_train.shape[1:]
    model = create_model(input_shape=input_shape, num_classes=10)
    
    # Print model summary
    model.summary()
    
    # Train model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    if save_model:
        print(f"Saving model to {model_path}...")
        model.save(model_path)
        print("Model saved successfully!")
    
    return model, history


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train digit classification model')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to synthetic digits dataset (default: auto-detect)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Fraction of data to use for validation')
    parser.add_argument('--model_path', type=str, default='models/digit_classifier.keras',
                       help='Path to save the trained model')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save the trained model')
    
    args = parser.parse_args()
    
    # Train the model
    model, history = train_model(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        save_model=not args.no_save,
        model_path=args.model_path
    )
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
