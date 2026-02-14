"""Hybrid Muon + AdamW optimizer for transformer models.

This example demonstrates how to use Muon optimizer for 2D weight matrices
(linear layers, embeddings) while using AdamW for 1D parameters (biases,
layer norms, etc.). This is a common pattern in large language models where:

- Muon: Optimizes weight matrices with orthogonalization (W, Q, K, V, O projections)
- AdamW: Optimizes biases and normalization parameters with decoupled weight decay

Benefits:
- Muon provides faster convergence for large matrices through orthogonalization
- AdamW handles 1D parameters which don't benefit from orthogonalization
- Memory efficient: no need to track per-param momentum for biases

Usage:
    python examples/mixed_adamw.py --model gpt2 --epochs 3 --lr 0.02

With FSDP2:
    torchrun --nproc_per_node=4 examples/mixed_adamw.py --model gpt2 --fsdp --epochs 3
"""

import argparse
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader, Dataset

from muon_fsdp import MuonOptimizer


class DummyDataset(Dataset):
    """Synthetic dataset for demonstration purposes."""

    def __init__(self, num_samples: int, seq_length: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.randint(1, self.vocab_size, (self.seq_length,))


class CausalSelfAttention(nn.Module):
    """Causal self-attention (simplified for demo)."""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        T_pos = att.size(-1)
        mask = (
            torch.triu(torch.ones(T_pos, T_pos, device=x.device, dtype=att.dtype), diagonal=1)
            * -1e10
        )
        att = att + mask

        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y


class TransformerBlock(nn.Module):
    """Transformer block with pre-LN architecture."""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    """GPT-style language model."""

    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.drop = nn.Dropout(dropout)

        self.h = nn.ModuleList([TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd)

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

        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pos_emb = self.wpe(positions)
        tok_emb = self.wte(input_ids)

        x = self.drop(tok_emb + pos_emb)

        for block in self.h:
            x = block(x, attention_mask)

        x = self.ln_f(x)

        lm_logits = x @ self.wte.weight.t()
        return lm_logits

    def configure_fsdp(self) -> None:
        """Configure FSDP2 for distributed training."""
        for module in self.h:
            fully_shard(module)


class HybridMuonAdamW(optim.Optimizer):
    """Hybrid optimizer combining Muon (for 2D matrices) and AdamW (for 1D params).

    This optimizer:
    - Uses Muon for 2D weight matrices (Linear layers, Embeddings)
    - Uses AdamW for 1D parameters (Biases, LayerNorm weights/biases)

    Benefits:
    - Faster convergence for large matrices through orthogonalization
    - Standard AdamW behavior for parameters that don't benefit from NS
    - Decoupled weight decay for both components
    - Memory efficient

    Args:
        params: Iterable of parameters to optimize.
        muon_lr: Learning rate for Muon (2D matrices). Default: 0.02
        adamw_lr: Learning rate for AdamW (1D params). Default: 1e-3
        muon_momentum: Momentum for Muon. Default: 0.95
        adamw_beta1: Beta1 for AdamW. Default: 0.9
        adamw_beta2: Beta2 for AdamW. Default: 0.999
        adamw_eps: Epsilon for AdamW. Default: 1e-8
        muon_weight_decay: Weight decay for Muon. Default: 0.0
        adamw_weight_decay: Weight decay for AdamW. Default: 0.01
        ns_steps: Newton-Schulz iterations for Muon. Default: 5
    """

    def __init__(
        self,
        params,
        muon_lr: float = 0.02,
        adamw_lr: float = 1e-3,
        muon_momentum: float = 0.95,
        adamw_beta1: float = 0.9,
        adamw_beta2: float = 0.999,
        adamw_eps: float = 1e-8,
        muon_weight_decay: float = 0.0,
        adamw_weight_decay: float = 0.01,
        ns_steps: int = 5,
    ):
        defaults = dict(
            muon_lr=muon_lr,
            adamw_lr=adamw_lr,
            muon_momentum=muon_momentum,
            adamw_beta1=adamw_beta1,
            adamw_beta2=adamw_beta2,
            adamw_eps=adamw_eps,
            muon_weight_decay=muon_weight_decay,
            adamw_weight_decay=adamw_weight_decay,
            ns_steps=ns_steps,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("HybridMuonAdamW does not support sparse gradients")

                # Classify parameter type
                is_2d = p.dim() == 2

                if is_2d:
                    self._muon_step(p, grad, group)
                else:
                    self._adamw_step(p, grad, group)

        return loss

    def _muon_step(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        group: Dict,
    ) -> None:
        """Perform Muon step for 2D parameter."""
        lr = group["muon_lr"]
        momentum = group["muon_momentum"]
        weight_decay = group["muon_weight_decay"]
        ns_steps = group["ns_steps"]

        state = self.state[p]

        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(p)

        buf = state["momentum_buffer"]
        buf.lerp_(grad, 1 - momentum)

        # Newton-Schulz orthogonalization
        update = buf.clone()
        from muon_fsdp.utils import zeropower_via_newtonschulz5

        update = zeropower_via_newtonschulz5(update, steps=ns_steps)

        # Learning rate scaling
        m, n = p.shape
        min_dim = min(m, n)
        max_dim = max(m, n)
        lr_scale = max(1.0, min_dim / max_dim) ** 0.5
        update.mul_(lr_scale)

        # Apply weight decay
        if weight_decay != 0:
            p.mul_(1 - lr * weight_decay)

        # Apply update
        p.add_(update, alpha=-lr)

    def _adamw_step(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        group: Dict,
    ) -> None:
        """Perform AdamW step for 1D parameter."""
        lr = group["adamw_lr"]
        beta1 = group["adamw_beta1"]
        beta2 = group["adamw_beta2"]
        eps = group["adamw_eps"]
        weight_decay = group["adamw_weight_decay"]

        state = self.state[p]

        if len(state) == 0:
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] = 0

        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        step = state["step"]
        state["step"] = step + 1

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        # Compute step size
        step_size = lr / bias_correction1
        bias_correction2_sqrt = bias_correction2**0.5

        # Decoupled weight decay
        if weight_decay != 0:
            p.mul_(1 - lr * weight_decay)

        # Compute update
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
        update = exp_avg / denom

        # Apply update
        p.add_(update, alpha=-step_size)


class HybridMuonAdamWOptimizer:
    """Wrapper that automatically separates 2D and 1D parameters."""

    def __init__(
        self,
        model: nn.Module,
        muon_lr: float = 0.02,
        adamw_lr: float = 1e-3,
        muon_momentum: float = 0.95,
        adamw_beta1: float = 0.9,
        adamw_beta2: float = 0.999,
        adamw_eps: float = 1e-8,
        muon_weight_decay: float = 0.0,
        adamw_weight_decay: float = 0.01,
        ns_steps: int = 5,
        fsdp_model: Optional[nn.Module] = None,
    ):
        self.model = model
        self.fsdp_model = fsdp_model

        # Separate parameters by type
        self._classify_parameters(model)

        # Create single optimizer that handles both types
        self.optimizer = HybridMuonAdamW(
            params=[
                {
                    "params": self.muon_params,
                    "lr": muon_lr,
                    "momentum": muon_momentum,
                    "weight_decay": muon_weight_decay,
                    "ns_steps": ns_steps,
                },
                {
                    "params": self.adamw_params,
                    "lr": adamw_lr,
                    "beta1": adamw_beta1,
                    "beta2": adamw_beta2,
                    "eps": adamw_eps,
                    "weight_decay": adamw_weight_decay,
                },
            ],
            muon_lr=muon_lr,
            adamw_lr=adamw_lr,
            muon_momentum=muon_momentum,
            adamw_beta1=adamw_beta1,
            adamw_beta2=adamw_beta2,
            adamw_eps=adamw_eps,
            muon_weight_decay=muon_weight_decay,
            adamw_weight_decay=adamw_weight_decay,
            ns_steps=ns_steps,
        )

    def _classify_parameters(self, model: nn.Module) -> None:
        """Classify parameters into 2D (Muon) and 1D (AdamW) groups."""
        self.muon_params: List[nn.Parameter] = []
        self.adamw_params: List[nn.Parameter] = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # 2D parameters: weight matrices in Linear and Embedding layers
            if p.dim() == 2:
                self.muon_params.append(p)
            # 1D parameters: biases and LayerNorm parameters
            else:
                self.adamw_params.append(p)

    def statistics(self) -> Dict[str, int]:
        """Return parameter statistics."""
        muon_count = sum(p.numel() for p in self.muon_params)
        adamw_count = sum(p.numel() for p in self.adamw_params)
        total = muon_count + adamw_count

        return {
            "muon_params": len(self.muon_params),
            "muon_parameters": muon_count,
            "adamw_params": len(self.adamw_params),
            "adamw_parameters": adamw_count,
            "total_params": total,
            "muon_ratio": muon_count / total * 100 if total > 0 else 0,
        }

    def step(self, closure=None) -> Optional[float]:
        return self.optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.optimizer.zero_grad(set_to_none)

    def state_dict(self) -> Dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict) -> None:
        self.optimizer.load_state_dict(state_dict)


def create_hybrid_optimizer(
    model: nn.Module,
    muon_lr: float = 0.02,
    adamw_lr: float = 1e-3,
    muon_weight_decay: float = 0.0,
    adamw_weight_decay: float = 0.01,
    use_fsdp: bool = False,
    fsdp_model: Optional[nn.Module] = None,
) -> HybridMuonAdamWOptimizer:
    """Create hybrid optimizer for transformer models.

    Args:
        model: The model to optimize.
        muon_lr: Learning rate for 2D matrices (default: 0.02).
        adamw_lr: Learning rate for 1D parameters (default: 1e-3).
        muon_weight_decay: Weight decay for Muon (default: 0.0).
        adamw_weight_decay: Weight decay for AdamW (default: 0.01).
        use_fsdp: Whether using FSDP2.
        fsdp_model: FSDP-wrapped model (required if use_fsdp=True).

    Returns:
        Hybrid optimizer wrapper.
    """
    return HybridMuonAdamWOptimizer(
        model=model,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        muon_weight_decay=muon_weight_decay,
        adamw_weight_decay=adamw_weight_decay,
        fsdp_model=fsdp_model if use_fsdp else None,
    )


class HybridTrainer:
    """Trainer for models with hybrid Muon + AdamW optimizer."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: HybridMuonAdamWOptimizer,
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

        self.scaler = ShardedGradScaler() if use_fsdp and use_mixed_precision else None
        self.global_step = 0
        self.epoch = 0

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            attention_mask = (batch != 0).float()

            # Forward pass
            try:
                from torch.cuda.amp import autocast

                with autocast():
                    logits = self.model(batch, attention_mask)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = batch[:, 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )
                    loss = loss / self.gradient_accumulation_steps
            except ImportError:
                logits = self.model(batch, attention_mask)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer.optimizer)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer.optimizer)
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


def main():
    parser = argparse.ArgumentParser(description="Train GPT with hybrid Muon+AdamW optimizer")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium"],
        help="Model size",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument(
        "--muon-lr", type=float, default=0.02, help="Muon learning rate for 2D matrices"
    )
    parser.add_argument(
        "--adamw-lr", type=float, default=1e-3, help="AdamW learning rate for 1D params"
    )
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP2 for distributed training")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    print(f"Training {args.model} with hybrid Muon + AdamW optimizer")
    print(f"Device: {args.device}")
    print(f"Muon LR: {args.muon_lr}")
    print(f"AdamW LR: {args.adamw_lr}")

    # Model configuration
    configs = {
        "gpt2": {"n_embd": 768, "n_layer": 12, "n_head": 12},
        "gpt2-medium": {"n_embd": 1024, "n_layer": 24, "n_head": 16},
    }
    config = configs[args.model]

    # Create model
    model = GPTModel(
        vocab_size=50257,
        n_positions=1024,
        n_embd=config["n_embd"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
    )
    model = model.to(args.device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    # Configure FSDP if requested
    if args.fsdp:
        model.configure_fsdp()
        fsdp_model = model
    else:
        fsdp_model = None

    # Create hybrid optimizer
    optimizer = create_hybrid_optimizer(
        model=model,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        muon_weight_decay=0.0,
        adamw_weight_decay=0.01,
        use_fsdp=args.fsdp,
        fsdp_model=fsdp_model,
    )

    # Print parameter statistics
    stats = optimizer.statistics()
    print(f"\nParameter distribution:")
    print(
        f"  Muon (2D): {stats['muon_params']} tensors, {stats['muon_parameters']:,} params "
        f"({stats['muon_ratio']:.1f}%)"
    )
    print(
        f"  AdamW (1D): {stats['adamw_params']} tensors, {stats['adamw_parameters']:,} params "
        f"({100 - stats['muon_ratio']:.1f}%)"
    )

    # Create dataset and dataloader
    num_samples = 1000
    train_dataset = DummyDataset(
        num_samples=num_samples,
        seq_length=args.seq_length,
        vocab_size=model.vocab_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(args.device == "cuda"),
        drop_last=True,
    )

    # Create trainer
    trainer = HybridTrainer(
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


if __name__ == "__main__":
    main()
