"""RCF state_dict/optimizer/lr_scheduler checkpoints, based on upstream train.py."""

from pathlib import Path
import torch


def save_training_checkpoint(path, epoch, model, optimizer, scheduler, metrics=None):
    """Save completed epoch count and the scheduler state for the next epoch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "metrics": metrics or {},
        "rcf_training_version": 1,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return path


def restore_training_checkpoint(path, model, optimizer, scheduler, map_location="cpu"):
    """Resume checkpoints written by this adaptation (one-based completed epochs)."""
    state = torch.load(Path(path), map_location=map_location, weights_only=True)
    if state.get("rcf_training_version") != 1:
        raise ValueError("Resume needs a local RCF training checkpoint; use original "
                         "weights as initial_checkpoint for fine-tuning")
    model.load_state_dict(state["state_dict"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["lr_scheduler"])
    return int(state["epoch"])
