import os
import sys
from pathlib import Path

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from data_generation.syntheic_digits import generate_sample_digits

def main():
    """Generate synthetic digit sample data for training."""
    print("Generating synthetic digit sample data...")
    
    # Configuration
    output_dir = '/data/synthetic_digits'
    n_samples = 1000
    font_size = 20
    
    # Generate samples
    generate_sample_digits(
        output_dir=output_dir,
        n_samples=n_samples,
        font_size=font_size
    )
    
    print(f"✓ Generated {n_samples} synthetic digit samples in {output_dir}/")
    print("Sample files created with format: digit_X_font_Y_sample_Z.png")

if __name__ == "__main__":
    main()
