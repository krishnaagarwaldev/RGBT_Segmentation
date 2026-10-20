"""Checkpoint management and Early Stopping utilities for model training."""
import os
import torch
from typing import Dict, Any, Optional


class CheckpointManager:
    """Manages saving and loading of model checkpoints."""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_metric = -float("inf")

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler],
        epoch: int,
        metrics: Dict[str, Any],
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> str:
        """Save a training checkpoint.

        Args:
            model: PyTorch model.
            optimizer: PyTorch optimizer.
            scaler: GradScaler for AMP.
            epoch: Current epoch index.
            metrics: Current epoch evaluation metrics dict.
            is_best: If True, also updates best_model.pth.
            filename: Optional custom filename.

        Returns:
            Path to saved checkpoint.
        """
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "metrics": metrics,
        }

        # Save last checkpoint
        last_path = os.path.join(self.save_dir, filename if filename else "last_model.pth")
        torch.save(state, last_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(state, best_path)

        return last_path

    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """Load a checkpoint into model and optimizer.

        Returns:
            Dict containing checkpoint metadata (epoch, metrics, etc.).
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        return checkpoint


class EarlyStopping:
    """Early stopping to stop training when monitored metric stops improving."""

    def __init__(self, patience: int = 15, delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, current_score: float) -> bool:
        """Update counter and check if early stopping should trigger.

        Returns:
            is_best: True if current score is the new best score.
        """
        if self.best_score is None:
            self.best_score = current_score
            return True

        if self.mode == "max":
            improved = current_score > self.best_score + self.delta
        else:
            improved = current_score < self.best_score - self.delta

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
