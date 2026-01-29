"""
train.py
Author: Lennin Abrão Sousa Santos
Data criação: 2026-01-29
Date update: 2026-01-29
Explicação: Training script for the SimpleCNN model.
How to use: uv run mlsc train
Licença: AGPL3
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from mlsc.dataset import ShapesDataset
from mlsc.model import SimpleCNN


def train():
    # Device config
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

    # Dataset
    # Note: ShapesDataset defaults to looking in ../data/raw relative to dataset.py
    # which is consistent with our structure
    full_dataset = ShapesDataset()

    if len(full_dataset) == 0:
        print("Error: No data found! Run generate_data.py first.")
        return

    # Split train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = SimpleCNN().to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training for {num_epochs} epochs...")

    # Train Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation Loop
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Acc: {val_acc:.2f}%"
        )

    # Save Model
    # Save in the root mlsc/ directory from where we likely run it, or relative to this script
    # User prompt just says "Salva o modelo final."
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")


if __name__ == "__main__":
    train()
