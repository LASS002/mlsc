"""
predict.py
Author: Lennin Abrão Sousa Santos
Data criação: 2026-02-02
Date update: 2026-02-02
Explicação: Performs inference on test images using trained model.
How to use: uv run mlsc predict --model <model_path> --data <data_dir>
Licença: AGPL3
"""

import torch
from pathlib import Path
import argparse
from PIL import Image
import torchvision.transforms as T
from mlsc.model import SimpleCNN
import csv


def predict_images(model_path, data_dir, output_csv=None):
    """
    Performs inference on images in data_dir using the trained model.
    
    Args:
        model_path: Path to saved model (.pth file)
        data_dir: Path to directory containing preprocessed images
        output_csv: Path to save results CSV (optional)
    
    Returns:
        Dictionary with results and metrics
    """
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    
    if not model_path.exists():
        raise ValueError(f"Model file {model_path} does not exist!")
    
    if not data_dir.exists():
        raise ValueError(f"Data directory {data_dir} does not exist!")
    
    # Device config
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    
    # Load model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"✓ Loaded model from {model_path}")
    
    # Transform (same as training)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    
    # Label mapping
    label_names = {0: "circle", 1: "square"}
    
    # Results storage
    results = []
    correct = 0
    total = 0
    
    # Confusion matrix [true_label][predicted_label]
    confusion = [[0, 0], [0, 0]]  # [[TN, FP], [FN, TP]]
    
    print("\nRunning predictions...")
    
    # Process circles (true label = 0)
    circle_dir = data_dir / "circle"
    if circle_dir.exists():
        for img_path in sorted(circle_dir.glob("*.png")):
            img = Image.open(img_path).convert("L")
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                pred_label = predicted.item()
            
            true_label = 0
            is_correct = (pred_label == true_label)
            
            results.append({
                "filename": img_path.name,
                "true_label": label_names[true_label],
                "predicted_label": label_names[pred_label],
                "correct": is_correct
            })
            
            confusion[true_label][pred_label] += 1
            if is_correct:
                correct += 1
            total += 1
    
    # Process squares (true label = 1)
    square_dir = data_dir / "square"
    if square_dir.exists():
        for img_path in sorted(square_dir.glob("*.png")):
            img = Image.open(img_path).convert("L")
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                pred_label = predicted.item()
            
            true_label = 1
            is_correct = (pred_label == true_label)
            
            results.append({
                "filename": img_path.name,
                "true_label": label_names[true_label],
                "predicted_label": label_names[pred_label],
                "correct": is_correct
            })
            
            confusion[true_label][pred_label] += 1
            if is_correct:
                correct += 1
            total += 1
    
    # Calculate metrics
    accuracy = 100 * correct / total if total > 0 else 0
    
    # Print results
    print("\n" + "="*60)
    print("RESULTADOS DA PREDIÇÃO")
    print("="*60)
    print(f"Total de imagens: {total}")
    print(f"Predições corretas: {correct}")
    print(f"Predições incorretas: {total - correct}")
    print(f"Acurácia: {accuracy:.2f}%")
    print("\nMatriz de Confusão:")
    print(f"                Predito: Circle  Predito: Square")
    print(f"Real: Circle         {confusion[0][0]:3d}            {confusion[0][1]:3d}")
    print(f"Real: Square         {confusion[1][0]:3d}            {confusion[1][1]:3d}")
    print("="*60)
    
    # Show some misclassified examples
    misclassified = [r for r in results if not r["correct"]]
    if misclassified:
        print(f"\nExemplos de Erros de Classificação ({len(misclassified)} total):")
        for i, res in enumerate(misclassified[:10], 1):  # Show first 10
            print(f"  {i}. {res['filename']}: Real={res['true_label']}, Predito={res['predicted_label']}")
        if len(misclassified) > 10:
            print(f"  ... e mais {len(misclassified) - 10} erros")
    
    
    # Save to CSV
    if output_csv is None:
        # Default: save to data/test/results_XXX.csv with auto-incrementing number
        test_dir = Path(__file__).parent.parent / "data" / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Find next available number
        existing_results = list(test_dir.glob("results_*.csv"))
        if existing_results:
            # Extract numbers from filenames like results_001.csv
            numbers = []
            for f in existing_results:
                try:
                    num = int(f.stem.split('_')[1])
                    numbers.append(num)
                except (IndexError, ValueError):
                    pass
            next_num = max(numbers) + 1 if numbers else 1
        else:
            next_num = 1
        
        output_csv = test_dir / f"results_{next_num:03d}.csv"
    else:
        output_csv = Path(output_csv)
    
    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "true_label", "predicted_label", "correct"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Resultados salvos em {output_csv}")
    
    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "confusion_matrix": confusion,
        "results": results
    }


def main():
    """CLI entry point for predict command."""
    parser = argparse.ArgumentParser(
        description="Run inference on test images using trained model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model.pth",
        help="Path to trained model file (default: model.pth)"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to preprocessed test data (with circle/ and square/ subdirs)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results CSV (default: auto-generate in data/test/results_XXX.csv)"
    )
    
    args = parser.parse_args()
    
    try:
        predict_images(args.model, args.data, args.output)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
