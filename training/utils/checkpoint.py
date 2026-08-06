import os
import glob
import signal
import torch


def find_latest_checkpoint(run_id, drive_base):
    """Return (path, epoch) of the latest checkpoint for run_id, or (None, 0)."""
    ckpt_dir = os.path.join(drive_base, "checkpoints", run_id)
    if not os.path.isdir(ckpt_dir):
        return None, 0

    candidates = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_epoch*.pt")))
    if not candidates:
        return None, 0

    latest = candidates[-1]
    try:
        epoch = int(os.path.basename(latest).replace("ckpt_epoch", "").replace(".pt", ""))
    except ValueError:
        epoch = 0
    return latest, epoch


def save_checkpoint(model, optimizer, epoch, run_id, drive_base):
    """Save model + optimizer state to Drive checkpoint directory."""
    ckpt_dir = os.path.join(drive_base, "checkpoints", run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch:03d}.pt")
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, ckpt_path)
    print(f"[AffectScore] Checkpoint saved: {ckpt_path}")


def install_sigterm_handler(run_id, model, optimizer, epoch_ref, drive_base):
    """Install SIGTERM handler that saves a checkpoint before the process exits."""
    def _handler(signum, frame):
        print(f"[AffectScore] SIGTERM -- saving emergency checkpoint (epoch {epoch_ref[0]})...")
        save_checkpoint(model, optimizer, epoch_ref[0], run_id, drive_base)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handler)
