# Sudoku Solver

A computer vision-based Sudoku solver that can detect and solve Sudoku puzzles from camera input.

## Features

- Real-time Sudoku grid detection from camera
- Manual digit collection for training data
- Synthetic data generation for data augmentation
- Digit classification using machine learning
- Complete Sudoku solving algorithm

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Activate virtual environment (if using one):
```bash
source venv/bin/activate
```

## Usage

### 1. Manual Digit Collection

Collect training data by manually labeling digits from captured Sudoku grids:

```bash
python run_manual_digit_collection.py
```

This will:
- Capture a Sudoku grid from your camera
- Allow you to manually assign digits to each cell
- Save the labeled images to `data/digits/manual/`

### 2. Synthetic Data Generation

Generate additional training data by creating augmented versions of manually collected images:

```bash
# Generate 100 synthetic samples per digit (default, append mode)
python generate_synthetic_data.py

# Generate 50 synthetic samples per digit in overwrite mode
python generate_synthetic_data.py --n-samples-per-digit 50 --mode overwrite

# Interactive mode
python run_synthetic_data_generation.py
```

The synthetic data generator applies various transformations:
- Perspective distortions (simulating camera angles)
- Shading variations (brightness, contrast, gamma)
- Rotation and scaling
- Noise and blur for robustness

### 3. Training the Model

Train a digit classifier on the collected data:

```bash
python -m src.training.train
```

### 4. Running the Solver

Run the complete Sudoku solver:

```bash
python main.py
```

## Data Structure

```
data/
├── digits/
│   ├── manual/          # Manually collected digit images
│   │   ├── 0_g0_c0.jpg     # digit_grid_cell.jpg
│   │   ├── 1_g0_c1.jpg
│   │   └── ...
│   └── synthetic/       # Generated synthetic images
│       ├── syn_0_g0_c0_000.jpg  # syn_digit_grid_cell_id.jpg
│       ├── syn_1_g0_c1_001.jpg
│       └── ...
```

### Naming Convention

**Manual Images**: `{digit}_g{grid_id}_c{cell_id}.jpg`
- `digit`: The digit value (0-9)
- `grid_id`: Grid identifier (0, 1, 2, ...)
- `cell_id`: Cell position in 9x9 grid (0-80)

**Synthetic Images**: `syn_{digit}_g{grid_id}_c{cell_id}_{synthetic_id}.jpg`
- `digit`: The digit value (0-9)
- `grid_id`: Grid identifier (0, 1, 2, ...)
- `cell_id`: Cell position in 9x9 grid (0-80)
- `synthetic_id`: Synthetic sample identifier (000, 001, 002, ...)

**Examples**:
- `5_g0_c23.jpg` - Manual image of digit 5 from grid 0, cell 23
- `syn_5_g0_c23_001.jpg` - Synthetic image of digit 5 from grid 0, cell 23, sample 1

## Project Structure

```
src/
├── capture/             # Video capture and frame processing
├── detection/           # Grid detection and cell extraction
├── training/            # Model training and dataset handling
└── scripts/             # Utility scripts
    ├── manual_digit_collection.py
    └── synethic_data_generator.py
```

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- TensorFlow/Keras
- Other dependencies listed in requirements.txt
