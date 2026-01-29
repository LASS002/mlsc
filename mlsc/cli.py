"""
cli.py
Author: Lennin Abrão Sousa Santos
Data criação: 2026-01-29
Date update: 2026-01-29
Explicação: Main entry point for the MLSC CLI.
How to use: uv run mlsc {generate|train}
Licença: AGPL3
"""
import argparse
import sys
from mlsc import generate_data
from mlsc import train


def main():
    parser = argparse.ArgumentParser(
        description="MLSC: Machine Learning Square vs Circle CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: generate
    subparsers.add_parser("generate", help="Generate synthetic data")

    # Subcommand: train
    subparsers.add_parser("train", help="Train the model")

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

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
