"""Command-line interface for muon-fsdp.

This module provides the CLI entry point for the muon-fsdp package.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    """Main entry point for the muon-fsdp CLI.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        prog="muon-fsdp",
        description="Muon optimizer with FSDP2 support for PyTorch",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show information about muon-fsdp")
    info_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check installation and dependencies")
    check_parser.add_argument("--full", action="store_true", help="Full diagnostic check")

    args = parser.parse_args()

    if args.command == "info":
        from muon_fsdp import __version__

        print(f"muon-fsdp version: {__version__}")
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {__import__('torch').__version__}")

        if args.verbose:
            print("\nAvailable components:")
            print("  - MuonOptimizer: Single-GPU optimizer")
            print("  - FSDPMuonOptimizer: Distributed optimizer")
            print("  - Newton-Schulz iteration")
            print("  - Distributed communication primitives")

        return 0

    if args.command == "check":
        try:
            import torch

            print(f"✓ PyTorch {torch.__version__} installed")
        except ImportError:
            print("✗ PyTorch not installed")
            return 1

        try:
            from muon_fsdp import FSDPMuonOptimizer, MuonOptimizer  # noqa: F401
            from muon_fsdp.distributed import all_gather_grads  # noqa: F401
            from muon_fsdp.utils import zeropower_via_newtonschulz5  # noqa: F401

            print("✓ muon-fsdp components imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import muon-fsdp: {e}")
            return 1

        if args.full:
            print("\nRunning quick validation...")
            import torch

            from muon_fsdp import MuonOptimizer

            model = torch.nn.Linear(128, 128)
            MuonOptimizer(model.parameters(), lr=0.01)
            print("✓ MuonOptimizer initialized successfully")

        print("\n✓ All checks passed!")
        return 0

    # No command specified
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
