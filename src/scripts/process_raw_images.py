#!/usr/bin/env python3
"""
Process Raw Images Script
Takes raw cell images and applies the updated cell_extractor processing,
saving the results to the processed folder after removing old processed images.
"""

import cv2 as cv
import numpy as np
import os
import sys
from pathlib import Path
import shutil

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.detection.cell_extactor import CellExtractor


class RawImageProcessor:
    def __init__(self):
        self.raw_dir = Path("data/digits/manual/raw")
        self.processed_dir = Path("data/digits/manual/processed")
        self.cell_extractor = None
        
    def setup_directories(self):
        """Ensure directories exist"""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def clear_processed_folder(self):
        """Remove all existing processed images"""
        print("Clearing processed folder...")
        if self.processed_dir.exists():
            for file in self.processed_dir.iterdir():
                if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    file.unlink()
        print("Processed folder cleared.")
        
    def get_raw_image_files(self):
        """Get all raw image files"""
        raw_files = []
        if self.raw_dir.exists():
            for file in self.raw_dir.iterdir():
                if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    raw_files.append(file)
        return sorted(raw_files)
    
    def create_dummy_grid(self, cell_image):
        """Create a dummy 9x9 grid with the cell image for CellExtractor"""
        # Create a 450x450 grid (9x9 cells of 50x50 each)
        grid = np.ones((450, 450), dtype=np.uint8) * 255
        
        # Place the cell image in the center (position 4,4)
        center_row, center_col = 4, 4
        start_y = center_row * 50
        end_y = start_y + 50
        start_x = center_col * 50
        end_x = start_x + 50
        
        # Resize cell image to 50x50 if needed
        if cell_image.shape != (50, 50):
            cell_resized = cv.resize(cell_image, (50, 50))
        else:
            cell_resized = cell_image
            
        grid[start_y:end_y, start_x:end_x] = cell_resized
        
        return grid
    
    def process_single_image(self, raw_file_path):
        """Process a single raw image"""
        try:
            # Load the raw image
            raw_image = cv.imread(str(raw_file_path), cv.IMREAD_GRAYSCALE)
            if raw_image is None:
                print(f"Failed to load image: {raw_file_path}")
                return False
                
            # Create a dummy grid with the cell image
            dummy_grid = self.create_dummy_grid(raw_image)
            
            # Initialize CellExtractor with the dummy grid
            cell_extractor = CellExtractor(dummy_grid)
            
            # Extract the processed cell (should be at position 4,4)
            processed_cells = cell_extractor.extract_cells()
            
            # Get the processed cell from the center position (index 40 = 4*9 + 4)
            if processed_cells[40] is not None:
                processed_cell = processed_cells[40]
                
                # Generate output filename
                raw_name = raw_file_path.stem
                # Replace '_raw' with '_processed'
                processed_name = raw_name.replace('_raw', '_processed') + '.jpg'
                output_path = self.processed_dir / processed_name
                
                # Save the processed image
                success = cv.imwrite(str(output_path), processed_cell)
                if success:
                    print(f"Processed: {raw_file_path.name} -> {processed_name}")
                    return True
                else:
                    print(f"Failed to save: {output_path}")
                    return False
            else:
                print(f"No processed cell found for: {raw_file_path.name}")
                return False
                
        except Exception as e:
            print(f"Error processing {raw_file_path.name}: {str(e)}")
            return False
    
    def process_all_images(self):
        """Process all raw images"""
        print("Starting image processing...")
        
        # Get all raw image files
        raw_files = self.get_raw_image_files()
        
        if not raw_files:
            print("No raw images found!")
            return
            
        print(f"Found {len(raw_files)} raw images to process")
        
        # Process each image
        success_count = 0
        total_count = len(raw_files)
        
        for i, raw_file in enumerate(raw_files, 1):
            print(f"Processing {i}/{total_count}: {raw_file.name}")
            if self.process_single_image(raw_file):
                success_count += 1
                
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {success_count}/{total_count} images")
        
    def run(self):
        """Main execution method"""
        print("Raw Image Processor")
        print("==================")
        
        # Setup directories
        self.setup_directories()
        
        # Clear processed folder
        self.clear_processed_folder()
        
        # Process all images
        self.process_all_images()


def main():
    """Main function"""
    processor = RawImageProcessor()
    processor.run()


if __name__ == "__main__":
    main()
