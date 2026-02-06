"""
organize_test_data.py
Author: Lennin Abrão Sousa Santos
Data criação: 2026-02-02
Date update: 2026-02-02
Explicação: Organizes hand-drawn test images into the correct directory structure.
How to use: uv run mlsc organize-test --source <source_dir>
Licença: AGPL3
"""

import shutil
from pathlib import Path
import argparse


def organize_test_data(source_dir, dest_dir=None):
    """
    Organizes test images from source directory into data/test structure.
    
    Args:
        source_dir: Path to directory containing test images
        dest_dir: Destination directory (defaults to data/test)
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        raise ValueError(f"Source directory {source_dir} does not exist!")
    
    # Default destination is data/test relative to project root
    if dest_dir is None:
        dest_dir = Path(__file__).parent.parent / "data" / "test"
    else:
        dest_dir = Path(dest_dir)
    
    # Create subdirectories
    circle_dir = dest_dir / "circle"
    square_dir = dest_dir / "square"
    
    circle_dir.mkdir(parents=True, exist_ok=True)
    square_dir.mkdir(parents=True, exist_ok=True)
    
    # Counters
    circles_copied = 0
    squares_copied = 0
    
    print(f"Organizing test data from {source_path} to {dest_dir}")
    
    # Process all PNG files in source directory
    for img_path in source_path.glob("*.png"):
        filename = img_path.name
        
        # Check if filename contains "_p" marker (can be prefix or suffix)
        has_p_marker = "_p" in filename.lower()
        
        # Identify circles (files starting with "circ")
        if filename.startswith("circ"):
            # Create standardized name with _p suffix if present
            if has_p_marker:
                new_name = f"circle_{circles_copied:03d}_p.png"
            else:
                new_name = f"circle_{circles_copied:03d}.png"
            dest_path = circle_dir / new_name
            shutil.copy2(img_path, dest_path)
            circles_copied += 1
            
        # Identify squares (files starting with "quad")
        elif filename.startswith("quad"):
            # Create standardized name with _p suffix if present
            if has_p_marker:
                new_name = f"square_{squares_copied:03d}_p.png"
            else:
                new_name = f"square_{squares_copied:03d}.png"
            dest_path = square_dir / new_name
            shutil.copy2(img_path, dest_path)
            squares_copied += 1
    
    print(f"✓ Copied {circles_copied} circle images to {circle_dir}")
    print(f"✓ Copied {squares_copied} square images to {square_dir}")
    print(f"✓ Total: {circles_copied + squares_copied} images organized")


def main():
    """CLI entry point for organize-test command."""
    parser = argparse.ArgumentParser(
        description="Organize hand-drawn test images into data/test structure"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory containing test images"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Destination directory (default: data/test)"
    )
    
    args = parser.parse_args()
    
    try:
        organize_test_data(args.source, args.dest)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
