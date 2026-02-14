"""Spectral Sphere Optimizer (SSO) training example.

This example demonstrates how to train a simple neural network using the
Spectral Sphere Optimizer (SSO). SSO extends Muon with spectral constraints,
providing better control over feature learning and improved training stability.

Key features:
- Spectral norm constraints on weight matrices
- Power iteration for spectral norm computation
- Comparison with standard Muon optimizer
- Visualization of spectral norm evolution

Usage:
    # Basic training with SSO
    python examples/train_sso.py --epochs 10 --lr 0.02

    # Compare SSO vs Muon
    python examples/train_sso.py --compare --epochs 10

    # Custom spectral constraint settings
    python examples/train_sso.py --radius-mode spectral_mup --retract-mode hard

    # Different model sizes
    python examples/train_sso.py --hidden-size 512 --epochs 20
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from muon_fsdp import MuonOptimizer
from muon_fsdp.sso import SpectralSphereOptimizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model with Spectral Sphere Optimizer")

    # Model configuration
    parser.add_argument(
        "--input-size",
        type=int,
        default=784,
        help="Input dimension (default: 784 for MNIST-like data)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Hidden layer size (default: 256)",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=10,
        help="Output dimension (default: 10 for classification)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of hidden layers (default: 3)",
    )

    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
        help="Learning rate (default: 0.02)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.95,
        help="Momentum coefficient (default: 0.95)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )

    # SSO-specific configuration
    parser.add_argument(
        "--power-iteration-steps",
        type=int,
        default=10,
        help="Power iteration steps for spectral norm (default: 10)",
    )
    parser.add_argument(
        "--radius-mode",
        type=str,
        default="spectral_mup",
        choices=["spectral_mup", "identity"],
        help="Target radius computation mode (default: spectral_mup)",
    )
    parser.add_argument(
        "--radius-scaler",
        type=float,
        default=1.0,
        help="Radius scaling factor (default: 1.0)",
    )
    parser.add_argument(
        "--retract-mode",
        type=str,
        default="hard",
        choices=["hard", "dynamic"],
        help="Retraction mode for spectral sphere (default: hard)",
    )

    # Comparison mode
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare SSO with standard Muon optimizer",
    )

    # Newton-Schulz configuration
    parser.add_argument(
        "--ns-steps",
        type=int,
        default=5,
        help="Number of Newton-Schulz iterations (default: 5)",
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto/cpu/cuda, default: auto)",
    )

    return parser.parse_args()


class SimpleMLP(nn.Module):
    """Simple multi-layer perceptron for demonstration."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def create_synthetic_dataset(input_size, output_size, num_samples=10000):
    """Create synthetic dataset for training."""
    torch.manual_seed(42)

    # Generate random data
    X = torch.randn(num_samples, input_size)

    # Generate labels based on a simple function
    # This creates a non-trivial classification problem
    weights = torch.randn(input_size, output_size)
    logits = X @ weights
    y = logits.argmax(dim=1)

    return TensorDataset(X, y)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def get_spectral_norms(model):
    """Compute spectral norms of all weight matrices."""
    spectral_norms = {}
    for name, param in model.named_parameters():
        if param.dim() == 2:
            # Compute spectral norm using SVD
            _, S, _ = torch.svd(param.data)
            spectral_norms[name] = S[0].item()
    return spectral_norms


def print_spectral_norms(spectral_norms, title="Spectral Norms"):
    """Print spectral norms in a formatted way."""
    print(f"\n{title}:")
    print("-" * 50)
    for name, sigma in spectral_norms.items():
        print(f"  {name:30s}: {sigma:.4f}")


def train_with_optimizer(
    model,
    train_loader,
    test_loader,
    optimizer,
    epochs,
    device,
    optimizer_name="Optimizer",
):
    """Train model with given optimizer and track metrics."""
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'=' * 60}")
    print(f"Training with {optimizer_name}")
    print(f"{'=' * 60}")

    # Initial spectral norms
    initial_norms = get_spectral_norms(model)
    print_spectral_norms(initial_norms, "Initial Spectral Norms")

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "spectral_norms": [],
    }

    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Track spectral norms
        current_norms = get_spectral_norms(model)
        history["spectral_norms"].append(current_norms)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch + 1:3d}/{epochs}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%"
        )

    elapsed_time = time.time() - start_time

    # Final spectral norms
    final_norms = get_spectral_norms(model)
    print_spectral_norms(final_norms, "Final Spectral Norms")

    print(f"\nTraining completed in {elapsed_time:.2f}s")

    return history


def compare_optimizers(args, device):
    """Compare SSO with standard Muon optimizer."""
    print("\n" + "=" * 60)
    print("Comparing SSO vs Muon Optimizer")
    print("=" * 60)

    # Create datasets
    train_dataset = create_synthetic_dataset(args.input_size, args.output_size, num_samples=8000)
    test_dataset = create_synthetic_dataset(args.input_size, args.output_size, num_samples=2000)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Train with Muon
    torch.manual_seed(args.seed)
    model_muon = SimpleMLP(args.input_size, args.hidden_size, args.output_size, args.num_layers).to(
        device
    )

    optimizer_muon = MuonOptimizer(
        model_muon.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        ns_steps=args.ns_steps,
    )

    history_muon = train_with_optimizer(
        model_muon,
        train_loader,
        test_loader,
        optimizer_muon,
        args.epochs,
        device,
        "Muon Optimizer",
    )

    # Train with SSO
    torch.manual_seed(args.seed)
    model_sso = SimpleMLP(args.input_size, args.hidden_size, args.output_size, args.num_layers).to(
        device
    )

    optimizer_sso = SpectralSphereOptimizer(
        model_sso.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        ns_steps=args.ns_steps,
        power_iteration_steps=args.power_iteration_steps,
        radius_mode=args.radius_mode,
        radius_scaler=args.radius_scaler,
        retract_mode=args.retract_mode,
    )

    history_sso = train_with_optimizer(
        model_sso,
        train_loader,
        test_loader,
        optimizer_sso,
        args.epochs,
        device,
        "Spectral Sphere Optimizer",
    )

    # Print comparison
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"{'Metric':<30} {'Muon':>12} {'SSO':>12}")
    print("-" * 60)
    print(
        f"{'Final Train Loss':<30} {history_muon['train_loss'][-1]:>12.4f} "
        f"{history_sso['train_loss'][-1]:>12.4f}"
    )
    print(
        f"{'Final Train Acc (%)':<30} {history_muon['train_acc'][-1]:>12.2f} "
        f"{history_sso['train_acc'][-1]:>12.2f}"
    )
    print(
        f"{'Final Test Loss':<30} {history_muon['test_loss'][-1]:>12.4f} "
        f"{history_sso['test_loss'][-1]:>12.4f}"
    )
    print(
        f"{'Final Test Acc (%)':<30} {history_muon['test_acc'][-1]:>12.2f} "
        f"{history_sso['test_acc'][-1]:>12.2f}"
    )

    # Spectral norm stability comparison
    print("\n" + "=" * 60)
    print("Spectral Norm Stability (coefficient of variation)")
    print("=" * 60)

    for name in history_muon["spectral_norms"][0].keys():
        muon_values = [norms[name] for norms in history_muon["spectral_norms"]]
        sso_values = [norms[name] for norms in history_sso["spectral_norms"]]

        muon_mean = sum(muon_values) / len(muon_values)
        sso_mean = sum(sso_values) / len(sso_values)

        muon_std = math.sqrt(sum((x - muon_mean) ** 2 for x in muon_values) / len(muon_values))
        sso_std = math.sqrt(sum((x - sso_mean) ** 2 for x in sso_values) / len(sso_values))

        muon_cv = muon_std / muon_mean if muon_mean > 0 else 0
        sso_cv = sso_std / sso_mean if sso_mean > 0 else 0

        print(f"{name:<30} {muon_cv:>12.4f} {sso_cv:>12.4f}")

    print("\nNote: Lower coefficient of variation indicates more stable spectral norms")


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Model: {args.num_layers}-layer MLP ({args.hidden_size} hidden units)")
    print(f"Dataset: Synthetic classification ({args.output_size} classes)")

    if args.compare:
        compare_optimizers(args, device)
    else:
        # Single optimizer training
        train_dataset = create_synthetic_dataset(
            args.input_size, args.output_size, num_samples=8000
        )
        test_dataset = create_synthetic_dataset(args.input_size, args.output_size, num_samples=2000)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = SimpleMLP(args.input_size, args.hidden_size, args.output_size, args.num_layers).to(
            device
        )

        optimizer = SpectralSphereOptimizer(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            ns_steps=args.ns_steps,
            power_iteration_steps=args.power_iteration_steps,
            radius_mode=args.radius_mode,
            radius_scaler=args.radius_scaler,
            retract_mode=args.retract_mode,
        )

        train_with_optimizer(
            model,
            train_loader,
            test_loader,
            optimizer,
            args.epochs,
            device,
            "Spectral Sphere Optimizer",
        )


if __name__ == "__main__":
    main()
