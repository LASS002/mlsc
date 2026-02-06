"""
preprocess.py
Author: Lennin Abrão Sousa Santos
Data criação: 2026-02-02
Date update: 2026-02-02
Explicação: Preprocesses images for inference (resize, normalize, etc).
How to use: uv run mlsc preprocess --input <input_dir> --output <output_dir>
Licença: AGPL3
"""

from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import argparse


def preprocess_images(input_dir, output_dir=None):
    """
    Preprocesses images from input directory and saves to output directory.
    
    Args:
        input_dir: Path to directory containing raw images (with circle/ and square/ subdirs)
        output_dir: Path to save processed images (defaults to data/processed)
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory {input_dir} does not exist!")
    
    # Default output is data/processed
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "processed"
    else:
        output_dir = Path(output_dir)
    
    # Create output subdirectories
    circle_out = output_dir / "circle"
    square_out = output_dir / "square"
    
    circle_out.mkdir(parents=True, exist_ok=True)
    square_out.mkdir(parents=True, exist_ok=True)
    
    # Define preprocessing transform (same as training)
    # Resize to 64x64, convert to grayscale, save as PNG
    transform = T.Compose([
        T.Resize((64, 64)),
        T.Grayscale(num_output_channels=1)
    ])
    
    processed_count = 0
    
    print(f"Preprocessing images from {input_path} to {output_dir}")
    
    # Process circles
    circle_in = input_path / "circle"
    if circle_in.exists():
        for img_path in circle_in.glob("*.png"):
            img = Image.open(img_path)
            img_processed = transform(img)
            
            # Save with same filename
            output_path = circle_out / img_path.name
            img_processed.save(output_path)
            processed_count += 1
    
    # Process squares
    square_in = input_path / "square"
    if square_in.exists():
        for img_path in square_in.glob("*.png"):
            img = Image.open(img_path)
            img_processed = transform(img)
            
            # Save with same filename
            output_path = square_out / img_path.name
            img_processed.save(output_path)
            processed_count += 1
    
    print(f"✓ Processed {processed_count} images")
    print(f"✓ Saved to {output_dir}")


def main():
    """CLI entry point for preprocess command."""
    parser = argparse.ArgumentParser(
        description="Preprocess images for inference (resize to 64x64, grayscale)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing circle/ and square/ subdirectories"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/processed)"
    )
    
    args = parser.parse_args()
    
    try:
        preprocess_images(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
