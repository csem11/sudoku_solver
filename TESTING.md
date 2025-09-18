# Sudoku Grid Testing

This document describes the comprehensive testing functionality for the Sudoku solver project.

## Quick Start

### Interactive Testing Mode
```bash
python test_grid.py
```

### Main Application with Testing
```bash
python main.py
# Choose option 2 for testing mode
```

## Features

### GridTester Class

The `GridTester` class provides comprehensive testing capabilities:

#### Interactive Testing
- **Live video feed** with real-time grid detection
- **Visual feedback** showing detected grids with colored overlays
- **Capture functionality** to analyze specific frames
- **Comprehensive analysis** of captured grids

#### Controls
- `'q'` - Quit the application
- `'c'` - Capture current frame and analyze grid
- `'s'` - Save current frame
- `'r'` - Reset test session

#### Output Organization
All test results are saved to `test_results/` directory:

```
test_results/
├── captured_frames/          # Original and transformed frames
│   ├── 20241218_143022_frame.jpg
│   └── 20241218_143022_transformed.jpg
├── processed_images/         # Image processing steps
│   ├── 20241218_143022_processed.jpg    # Main processed image
│   ├── 20241218_143022_gray.jpg         # Grayscale conversion
│   ├── 20241218_143022_blurred.jpg      # Gaussian blur
│   ├── 20241218_143022_threshold.jpg    # Adaptive threshold
│   ├── 20241218_143022_edges.jpg        # Canny edge detection
│   └── 20241218_143022_contours.jpg     # Detected contours
├── extracted_cells/          # Individual cell images
│   └── 20241218_143022/
│       ├── cell_00_00.jpg    # Row 0, Col 0
│       ├── cell_00_01.jpg    # Row 0, Col 1
│       └── ...
├── processed_cells/          # Processed cell images
│   └── 20241218_143022/
│       ├── cell_00_00_final.jpg       # Final processed cell (28x28)
│       ├── cell_00_00_composite.jpg   # Labeled composite view
│       └── ...
└── predicted_grids/          # Prediction results
    └── 20241218_143022/
        ├── predicted_grid.txt
        ├── confidence_grid.txt
        ├── grid.npy
        └── probabilities.npy
```

## Image Processing Details

### Processed Images
The testing system saves multiple versions of processed images for debugging and analysis:

- **processed.jpg**: Main processed image used for grid detection
- **gray.jpg**: Grayscale conversion of the original image
- **blurred.jpg**: Gaussian blur applied to reduce noise
- **threshold.jpg**: Adaptive threshold for binary image
- **edges.jpg**: Canny edge detection results
- **contours.jpg**: All detected contours overlaid on original image

These images help understand how the grid detection pipeline processes the input and can be useful for debugging detection issues.

## Cell Processing Details

### Processed Cell Images
For each extracted cell, the final processed version is saved:

- **final.jpg**: Final processed cell (28x28 pixels, ready for model input)
- **composite.jpg**: Labeled composite view showing the processed cell

The cells are automatically preprocessed by the `CellExtractor.preprocess_cell()` method, which includes:
- Cropping (10% border removal)
- Upscaling (5x)
- Grayscale conversion
- Gaussian blur
- OTSU thresholding
- Inversion and morphological operations
- Final resize to 28x28 with centering

This preprocessing pipeline ensures optimal input format for the digit classification model.

## Cell Extraction Details

### Individual Cell Images
- Each cell is saved as a separate image file
- Filename format: `cell_{row:02d}_{col:02d}.jpg`
- Both original size and enlarged versions are saved
- Empty cells (None) are logged but not saved

### Cell Organization
- Cells are organized in a 9x9 grid
- Row and column indices start from 0
- Position (0,0) is top-left corner
- Position (8,8) is bottom-right corner

## Prediction Results

### Text Output
- **predicted_grid.txt**: Human-readable grid with digits and dots for empty cells
- **confidence_grid.txt**: Confidence scores for each prediction

### Binary Output
- **grid.npy**: NumPy array of predicted digits
- **probabilities.npy**: NumPy array of confidence scores

## Usage Examples

### Basic Interactive Testing
```python
from src import GridTester

tester = GridTester()
tester.run_interactive_test()
```

### Batch Testing
```python
from src import GridTester

tester = GridTester()
tester.run_batch_test("path/to/test/images")
```

## Troubleshooting

### Common Issues

1. **No grid detected**: Ensure good lighting and clear grid boundaries
2. **Wrong number of cells**: Check that the grid is properly transformed
3. **Low prediction confidence**: Verify cell images are clear and properly sized

### Debug Information

The tester provides detailed console output:
- Grid detection status
- Cell extraction counts
- Prediction success/failure
- File save locations

## Integration

The testing functionality is fully integrated with the main application:

- Import: `from src import GridTester`
- All detection, transformation, and generation modules are used
- Results are automatically organized and saved
- Visual feedback is provided throughout the process
