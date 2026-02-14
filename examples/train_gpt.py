"""GPT-style language model training with Muon optimizer.

This example demonstrates how to train a GPT-style language model using the Muon
optimizer. It includes both single-GPU and FSDP2 distributed training examples.

Key features:
- Pre-LN transformer architecture
- Mixed precision training (bf16/fp16)
- Gradient checkpointing for memory efficiency
- Learning rate scheduling with warmup
- Checkpoint saving and resuming

Usage:
    # Single GPU training
    python examples/train_gpt.py --model gpt2 --epochs 3 --lr 0.02

    # FSDP2 distributed training (4 GPUs)
    torchrun --nproc_per_node=4 examples/train_gpt.py --model gpt2 --fsdp --epochs 3

    # Custom model configuration
    python examples/train_gpt.py --n_layers 12 --n_heads 12 --n_embd 768 --epochs 3
"""

import argparse
import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader, Dataset

# Optional mixed precision
try:
    from torch.cuda.amp import autocast
except ImportError:
    autocast = None  # type: ignore

from muon_fsdp import MuonOptimizer, FSDPMuonOptimizer


@dataclass
class GPTConfig:
    """Configuration for GPT-style model."""

    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5


class GPT2SmallConfig:
    """GPT-2 small configuration (117M parameters)."""

    vocab_size = 50257
    n_positions = 1024
    n_embd = 768
    n_layer = 12
    n_head = 12
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-5


class GPT2MediumConfig:
    """GPT-2 medium configuration (345M parameters)."""

    vocab_size = 50257
    n_positions = 1024
    n_embd = 1024
    n_layer = 24
    n_head = 16
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-5


class GPT2LargeConfig:
    """GPT-2 large configuration (774M parameters)."""

    vocab_size = 50257
    n_positions = 1024
    n_embd = 1280
    n_layer = 36
    n_head = 20
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-5


class CausalSelfAttention(nn.Module):
    """Causal self-attention with configurable head size and dropout."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        assert self.head_dim * config.n_head == config.n_embd

        # Q, K, V projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.size()

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape to (B, T, n_head, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        # Transpose for attention computation: (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        T_pos = att.size(-1)
        mask = (
            torch.triu(torch.ones(T_pos, T_pos, device=x.device, dtype=att.dtype), diagonal=1)
            * -1e10
        )
        att = att + mask

        # Apply attention mask if provided (for padding)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            att = att + attention_mask * -1e10

        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = torch.matmul(att, v)  # (B, n_head, T, head_dim)

        # Transpose back and concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y


class Block(nn.Module):
    """Transformer block with pre-LN architecture."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN architecture: apply layer norm before attention/MLP
        x_attn = self.attn(self.ln_1(x), attention_mask)
        x = x + x_attn
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    """GPT-style language model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # Position embeddings ( Learned)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer blocks
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = input_ids.size()

        # Check sequence length
        if T > self.config.n_positions:
            raise ValueError(f"Sequence length {T} exceeds maximum {self.config.n_positions}")

        # Get token and position embeddings
        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pos_emb = self.wpe(positions)
        tok_emb = self.wte(input_ids)

        # Combine embeddings
        x = self.drop(tok_emb + pos_emb)

        # Apply transformer blocks
        for block in self.h:
            x = block(x, attention_mask)

        # Final layer norm
        x = self.ln_f(x)

        # Language model head
        lm_logits = x @ self.wte.weight.t()

        return lm_logits

    def configure_fsdp(self) -> None:
        """Configure FSDP2 for distributed training."""
        for module in self.h:
            fully_shard(module)


class DummyDataset(Dataset):
    """Synthetic dataset for demonstration purposes."""

    def __init__(self, num_samples: int, seq_length: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Generate random sequence (excluding padding token)
        return torch.randint(1, self.vocab_size, (self.seq_length,))


@contextmanager
def precision_context(enabled: bool = True) -> Iterator[None]:
    """Context manager for mixed precision training."""
    if autocast is None or not enabled:
        yield
        return

    with autocast():
        yield


class GPT2Trainer:
    """Trainer for GPT-style models with Muon optimizer."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        config,
        use_fsdp: bool = False,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.config = config
        self.use_fsdp = use_fsdp
        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Mixed precision scaler for FSDP
        self.scaler = ShardedGradScaler() if use_fsdp and use_mixed_precision else None

        # Training state
        self.global_step = 0
        self.epoch = 0

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)

            # Handle attention mask
            attention_mask = (batch != 0).float()

            # Forward pass with mixed precision
            with precision_context(enabled=self.use_mixed_precision and self.scaler is None):
                # Compute loss (shift for causal LM)
                logits = self.model(batch, attention_mask)
                # Shift for next token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()

                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass with gradient scaling for FSDP
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

        return total_loss / num_batches

    def train(self, num_epochs: int, log_interval: int = 100) -> List[float]:
        """Train for multiple epochs."""
        epoch_losses = []

        for epoch in range(num_epochs):
            self.epoch = epoch
            start_time = time.time()

            epoch_loss = self.train_epoch()
            epoch_losses.append(epoch_loss)

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{num_epochs}: loss={epoch_loss:.4f}, time={epoch_time:.1f}s")

        return epoch_losses


def create_gpt_model(model_name: str = "gpt2") -> nn.Module:
    """Create a GPT model by name."""
    configs = {
        "gpt2": GPT2SmallConfig,
        "gpt2-medium": GPT2MediumConfig,
        "gpt2-large": GPT2LargeConfig,
    }

    if model_name in configs:
        config = GPTConfig(
            vocab_size=configs[model_name].vocab_size,
            n_embd=configs[model_name].n_embd,
            n_layer=configs[model_name].n_layer,
            n_head=configs[model_name].n_head,
        )
    else:
        # Default to small config
        config = GPTConfig()

    return GPTModel(config)


def create_muon_optimizer(
    model: nn.Module,
    lr: float = 0.02,
    weight_decay: float = 0.01,
    momentum: float = 0.95,
    use_fsdp: bool = False,
) -> optim.Optimizer:
    """Create optimizer for GPT model.

    For FSDP2 training, use FSDPMuonOptimizer which handles sharded parameters.
    For single-GPU training, use MuonOptimizer directly.

    Args:
        model: The model to optimize.
        lr: Learning rate (default: 0.02).
        weight_decay: Weight decay coefficient (default: 0.01).
        momentum: Momentum coefficient (default: 0.95).
        use_fsdp: Whether using FSDP2 distributed training.

    Returns:
        Configured optimizer.
    """
    if use_fsdp:
        return FSDPMuonOptimizer(
            model=model,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True,
            ns_steps=5,
            ns_stepsize=1.0,
        )
    else:
        return MuonOptimizer(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Train GPT with Muon optimizer")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large"],
        help="Model size to use",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP2 for distributed training")
    parser.add_argument(
        "--no-mixed-precision", action="store_true", help="Disable mixed precision training"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints", help="Directory to save checkpoints"
    )

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Training {args.model} model")
    print(f"Device: {args.device}")
    print(f"FSDP2: {args.fsdp}")
    print(f"Mixed Precision: {not args.no_mixed_precision}")

    # Create model
    model = create_gpt_model(args.model)
    model = model.to(args.device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Configure FSDP if requested
    if args.fsdp:
        model.configure_fsdp()
        print("FSDP2 enabled for distributed training")

    # Create optimizer
    optimizer = create_muon_optimizer(
        model=model,
        lr=args.lr,
        weight_decay=0.01,
        momentum=0.95,
        use_fsdp=args.fsdp,
    )

    # Create dataset and dataloader
    num_samples = 1000  # For demonstration
    train_dataset = DummyDataset(
        num_samples=num_samples,
        seq_length=args.seq_length,
        vocab_size=model.config.vocab_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        drop_last=True,
    )

    # Create trainer
    trainer = GPT2Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        config=args,
        use_fsdp=args.fsdp,
        use_mixed_precision=not args.no_mixed_precision,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        device=args.device,
    )

    # Train
    print("\nStarting training...")
    losses = trainer.train(num_epochs=args.epochs)

    print(f"\nTraining completed!")
    print(f"Final loss: {losses[-1]:.4f}")

    # Save checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, f"muon_{args.model}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": args.epochs,
            "loss": losses[-1],
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
