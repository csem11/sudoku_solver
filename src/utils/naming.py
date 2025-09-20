#!/usr/bin/env python3
"""
Utility functions for consistent image naming conventions
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any


def parse_manual_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse manual image filename to extract metadata.
    
    Format: {digit}_g{grid_id}_c{cell_id}_man_{image_type}.jpg
    Example: "5_g0_c23_man_processed.jpg" -> {"digit": 5, "grid_id": 0, "cell_id": 23, "type": "manual", "image_type": "processed"}
    
    Args:
        filename: The filename to parse
        
    Returns:
        Dictionary with parsed metadata or None if invalid format
    """
    try:
        # Remove extension
        name = Path(filename).stem
        
        # Split by underscore
        parts = name.split('_')
        if len(parts) != 5 or parts[3] != 'man':
            return None
            
        digit, grid_part, cell_part, man_part, image_type = parts
        
        # Parse digit
        digit_val = int(digit)
        
        # Parse grid_id (remove 'g' prefix)
        if not grid_part.startswith('g'):
            return None
        grid_id = int(grid_part[1:])
        
        # Parse cell_id (remove 'c' prefix)
        if not cell_part.startswith('c'):
            return None
        cell_id = int(cell_part[1:])
        
        return {
            "digit": digit_val,
            "grid_id": grid_id,
            "cell_id": cell_id,
            "type": "manual",
            "image_type": image_type,
            "filename": filename
        }
        
    except (ValueError, IndexError):
        return None


def parse_synthetic_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse synthetic image filename to extract metadata.
    
    Format: {digit}_g{grid_id}_c{cell_id}_{synthetic_id}_syn.jpg
    Example: "5_g0_c23_001_syn.jpg" -> {"digit": 5, "grid_id": 0, "cell_id": 23, "synthetic_id": 1, "type": "synthetic"}
    
    Args:
        filename: The filename to parse
        
    Returns:
        Dictionary with parsed metadata or None if invalid format
    """
    try:
        # Remove extension
        name = Path(filename).stem
        
        # Split by underscore
        parts = name.split('_')
        if len(parts) != 5 or parts[4] != 'syn':
            return None
            
        digit, grid_part, cell_part, synthetic_id, syn_part = parts
        
        # Parse digit
        digit_val = int(digit)
        
        # Parse grid_id (remove 'g' prefix)
        if not grid_part.startswith('g'):
            return None
        grid_id = int(grid_part[1:])
        
        # Parse cell_id (remove 'c' prefix)
        if not cell_part.startswith('c'):
            return None
        cell_id = int(cell_part[1:])
        
        # Parse synthetic_id
        synthetic_id_val = int(synthetic_id)
        
        return {
            "digit": digit_val,
            "grid_id": grid_id,
            "cell_id": cell_id,
            "synthetic_id": synthetic_id_val,
            "type": "synthetic",
            "filename": filename
        }
        
    except (ValueError, IndexError):
        return None


def parse_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse any image filename (manual or synthetic) to extract metadata.
    
    Args:
        filename: The filename to parse
        
    Returns:
        Dictionary with parsed metadata or None if invalid format
    """
    # Try synthetic first (more specific)
    result = parse_synthetic_filename(filename)
    if result:
        return result
    
    # Try manual
    return parse_manual_filename(filename)


def generate_manual_filename(digit: int, grid_id: int, cell_id: int, image_type: str = "processed") -> str:
    """
    Generate manual image filename.
    
    Format: {digit}_g{grid_id}_c{cell_id}_man_{image_type}.jpg
    
    Args:
        digit: The digit value (0-9)
        grid_id: The grid identifier
        cell_id: The cell identifier
        image_type: Type of image ("raw" or "processed")
        
    Returns:
        Generated filename
    """
    return f"{digit}_g{grid_id}_c{cell_id}_man_{image_type}.jpg"


def generate_synthetic_filename(digit: int, grid_id: int, cell_id: int, synthetic_id: int) -> str:
    """
    Generate synthetic image filename.
    
    Format: {digit}_g{grid_id}_c{cell_id}_{synthetic_id}_syn.jpg
    
    Args:
        digit: The digit value (0-9)
        grid_id: The grid identifier
        cell_id: The cell identifier
        synthetic_id: The synthetic sample identifier
        
    Returns:
        Generated filename
    """
    return f"{digit}_g{grid_id}_c{cell_id}_{synthetic_id:03d}_syn.jpg"


def get_cell_position(cell_id: int) -> Tuple[int, int]:
    """
    Convert cell_id to row, col position in 9x9 grid.
    
    Args:
        cell_id: Cell identifier (0-80)
        
    Returns:
        Tuple of (row, col) coordinates
    """
    row = cell_id // 9
    col = cell_id % 9
    return row, col


def get_cell_id(row: int, col: int) -> int:
    """
    Convert row, col position to cell_id in 9x9 grid.
    
    Args:
        row: Row index (0-8)
        col: Column index (0-8)
        
    Returns:
        Cell identifier (0-80)
    """
    return row * 9 + col


def validate_filename(filename: str) -> bool:
    """
    Validate if filename follows the expected naming convention.
    
    Args:
        filename: The filename to validate
        
    Returns:
        True if valid, False otherwise
    """
    return parse_filename(filename) is not None


# Example usage and testing
if __name__ == "__main__":
    # Test manual filename parsing
    test_manual_raw = "5_g0_c23_man_raw.jpg"
    result = parse_manual_filename(test_manual_raw)
    print(f"Manual raw: {test_manual_raw} -> {result}")
    
    test_manual_processed = "5_g0_c23_man_processed.jpg"
    result = parse_manual_filename(test_manual_processed)
    print(f"Manual processed: {test_manual_processed} -> {result}")
    
    # Test synthetic filename parsing
    test_synthetic = "5_g0_c23_001_syn.jpg"
    result = parse_synthetic_filename(test_synthetic)
    print(f"Synthetic: {test_synthetic} -> {result}")
    
    # Test generation
    manual_raw_name = generate_manual_filename(5, 0, 23, "raw")
    print(f"Generated manual raw: {manual_raw_name}")
    
    manual_processed_name = generate_manual_filename(5, 0, 23, "processed")
    print(f"Generated manual processed: {manual_processed_name}")
    
    synthetic_name = generate_synthetic_filename(5, 0, 23, 1)
    print(f"Generated synthetic: {synthetic_name}")
    
    # Test cell position conversion
    row, col = get_cell_position(23)
    print(f"Cell 23 -> Row {row}, Col {col}")
    
    cell_id = get_cell_id(2, 5)
    print(f"Row 2, Col 5 -> Cell {cell_id}")
