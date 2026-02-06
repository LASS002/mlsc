"""
add_test_to_dataset.py
Author: Lennin Abrão Sousa Santos
Data criação: 2026-02-06
Date update: 2026-02-06
Explicação: Processes test images and adds them to the training dataset with '_m' suffix
How to use: uv run python mlsc/add_test_to_dataset.py
Licença: AGPL3
"""

from pathlib import Path
from PIL import Image
import torchvision.transforms as T


def add_test_images_to_dataset():
    """
    Processes images from data/test and adds them to dataset/ with '_m' suffix.
    
    The images are:
    1. Resized to 64x64
    2. Converted to grayscale
    3. Renamed with '_m' suffix (before extension)
    4. Saved to dataset/ directory
    """
    # Define paths
    base_dir = Path(__file__).parent.parent
    test_dir = base_dir / "data" / "test"
    dataset_dir = base_dir / "dataset"
    
    # Verify directories exist
    if not test_dir.exists():
        raise ValueError(f"Test directory {test_dir} does not exist!")
    
    # Create dataset directory if it doesn't exist
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Define preprocessing transform (same as training)
    # Resize to 64x64, convert to grayscale
    transform = T.Compose([
        T.Resize((64, 64)),
        T.Grayscale(num_output_channels=1)
    ])
    
    processed_count = 0
    
    print(f"Processing test images from {test_dir}")
    print(f"Adding to dataset: {dataset_dir}")
    print("-" * 60)
    
    # Process both circle and square subdirectories
    for class_name in ["circle", "square"]:
        class_dir = test_dir / class_name
        
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {class_name} images...")
        
        # Process each image in the subdirectory
        for img_path in sorted(class_dir.glob("*.png")):
            # Load and transform image
            img = Image.open(img_path)
            img_processed = transform(img)
            
            # Create new filename with '_m' suffix
            # e.g., "circle_001.png" -> "circle_001_m.png"
            original_name = img_path.stem  # filename without extension
            new_name = f"{original_name}_m.png"
            
            # Save to dataset directory
            output_path = dataset_dir / new_name
            img_processed.save(output_path)
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count} images...")
    
    print("-" * 60)
    print(f"✓ Successfully processed {processed_count} images")
    print(f"✓ Images saved to {dataset_dir}")
    print("\nNext steps:")
    print("  1. Verify the images in the dataset directory")
    print("  2. Retrain the model with: uv run mlsc train")


if __name__ == "__main__":
    try:
        add_test_images_to_dataset()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
