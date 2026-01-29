"""
generate_data.py
Author: Lennin Abrão Sousa Santos
Data criação: 2026-01-29
Date update: 2026-01-29
Explicação: Generates synthetic dataset of squares and circles.
How to use: uv run mlsc generate
Licença: AGPL3
"""

from PIL import Image, ImageDraw
import os
import random
from pathlib import Path


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_square(size=64):
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)

    # Random size for square
    square_size = random.randint(10, 40)

    # Random position
    x0 = random.randint(0, size - square_size)
    y0 = random.randint(0, size - square_size)
    x1 = x0 + square_size
    y1 = y0 + square_size

    draw.rectangle([x0, y0, x1, y1], fill=255)
    return img


def generate_circle(size=64):
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)

    # Random radius
    radius = random.randint(5, 20)

    # Random position (ensure circle fits)
    x_center = random.randint(radius, size - radius)
    y_center = random.randint(radius, size - radius)

    draw.ellipse(
        [x_center - radius, y_center - radius, x_center + radius, y_center + radius],
        fill=255,
    )
    return img


def main():
    # Define paths relative to this script
    current_dir = Path(__file__).parent
    raw_data_dir = current_dir.parent / "data" / "raw"

    square_dir = raw_data_dir / "square"
    circle_dir = raw_data_dir / "circle"

    create_directory(square_dir)
    create_directory(circle_dir)

    print("Generating squares...")
    for i in range(1000):
        img = generate_square()
        img.save(square_dir / f"square_{i}.png")

    print("Generating circles...")
    for i in range(1000):
        img = generate_circle()
        img.save(circle_dir / f"circle_{i}.png")

    print(f"Data generation complete. Saved to {raw_data_dir}")


if __name__ == "__main__":
    main()
