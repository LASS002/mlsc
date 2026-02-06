"""
cli.py
Author: Lennin Abrão Sousa Santos
Data criação: 2026-01-29
Date update: 2026-02-02
Explicação: Main entry point for the MLSC CLI.
How to use: uv run mlsc {generate|train|organize-test|preprocess|predict}
Licença: AGPL3
"""
import argparse
import sys
from mlsc import generate_data
from mlsc import train
from mlsc import organize_test_data
from mlsc import preprocess
from mlsc import predict


def main():
    parser = argparse.ArgumentParser(
        description="MLSC: Machine Learning Square vs Circle CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: generate
    subparsers.add_parser("generate", help="Generate synthetic data")

    # Subcommand: train
    subparsers.add_parser("train", help="Train the model")

    # Subcommand: organize-test
    organize_parser = subparsers.add_parser(
        "organize-test", help="Organize hand-drawn test images"
    )
    organize_parser.add_argument(
        "--source", type=str, required=True, help="Source directory with test images"
    )
    organize_parser.add_argument(
        "--dest", type=str, default=None, help="Destination directory (default: data/test)"
    )

    # Subcommand: preprocess
    preprocess_parser = subparsers.add_parser(
        "preprocess", help="Preprocess images for inference"
    )
    preprocess_parser.add_argument(
        "--input", type=str, required=True, help="Input directory with raw images"
    )
    preprocess_parser.add_argument(
        "--output", type=str, default=None, help="Output directory (default: data/processed)"
    )

    # Subcommand: predict
    predict_parser = subparsers.add_parser(
        "predict", help="Run inference on test images"
    )
    predict_parser.add_argument(
        "--model", type=str, default="model.pth", help="Path to trained model (default: model.pth)"
    )
    predict_parser.add_argument(
        "--data", type=str, required=True, help="Path to preprocessed test data"
    )
    predict_parser.add_argument(
        "--output", type=str, default=None, help="Path to save results CSV (default: auto-generated)"
    )

    args = parser.parse_args()

    if args.command == "generate":
        print("Starting data generation...")
        try:
            generate_data.main()
        except Exception as e:
            print(f"Error during data generation: {e}")
            sys.exit(1)

    elif args.command == "train":
        print("Starting model training...")
        try:
            train.train()
        except Exception as e:
            print(f"Error during training: {e}")
            sys.exit(1)

    elif args.command == "organize-test":
        print("Organizing test data...")
        try:
            organize_test_data.organize_test_data(args.source, args.dest)
        except Exception as e:
            print(f"Error organizing test data: {e}")
            sys.exit(1)

    elif args.command == "preprocess":
        print("Preprocessing images...")
        try:
            preprocess.preprocess_images(args.input, args.output)
        except Exception as e:
            print(f"Error preprocessing images: {e}")
            sys.exit(1)

    elif args.command == "predict":
        print("Running predictions...")
        try:
            predict.predict_images(args.model, args.data, args.output)
        except Exception as e:
            print(f"Error during prediction: {e}")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
