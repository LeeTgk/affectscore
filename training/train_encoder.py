"""Train the AffectEncoderMLP on MERT-annotated V-A pairs.

Two-layer MLP with GELU: input=6, hidden=64, output=512, L2-normalized.
Dropout=0.3 applied after the first activation during training.
"""
import os
import torch
import torch.nn as nn


class AffectEncoderMLP(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 512),
        )

    def forward(self, x):
        return nn.functional.normalize(self.net(x), p=2, dim=-1)


def _build_model(hidden_dim=64, dropout=0.3):
    return AffectEncoderMLP(hidden_dim=hidden_dim, dropout=dropout)


def train(X, Y, epochs=300, patience=30, output_path="encoder.pt"):
    """Train on (X, Y) with cosine-distance loss and early stopping.

    Args:
        X: np.ndarray (N, 6)   -- 6-d affect signals
        Y: np.ndarray (N, 512) -- target unit-sphere embeddings
        epochs:  max training epochs
        patience: early stopping patience
        output_path: where to save the best checkpoint

    Returns:
        (model, history) where history is a list of (epoch, loss) tuples.
    """
    model = _build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    history = []

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = (1.0 - (pred * Y_t).sum(dim=-1)).mean()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        history.append((epoch, loss_val))

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0:
            print(f"[AffectScore] Encoder epoch {epoch}/{epochs}: loss={loss_val:.6f}")

        if no_improve >= patience:
            print(f"[AffectScore] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"[AffectScore] Encoder saved: {output_path} (loss={best_loss:.6f})")
    return model, history


if __name__ == "__main__":
    import argparse
    import json
    import numpy as np

    parser = argparse.ArgumentParser(description="Train AffectEncoderMLP")
    parser.add_argument("--manifest", required=True, help="Path to training_set_clean_clap.json")
    parser.add_argument("--output",   default="training/encoder.pt")
    parser.add_argument("--epochs",   type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        records = json.load(f)

    # arc_position=0.5 (mid-arc default); engagement signals=0.0 (not in clip-level data)
    X = np.array(
        [[r["V_A_valence"], r["V_A_arousal"], 0.5, 0.0, 0.0, 0.0] for r in records],
        dtype=np.float32,
    )

    n = len(records)
    valence = X[:, 0].astype(np.float64)
    arousal = X[:, 1].astype(np.float64)
    theta = np.pi * (valence + 1.0) / 2.0
    phi = (np.pi / 2.0) * arousal
    target = np.zeros((n, 512), dtype=np.float64)
    target[:, 0] = np.cos(theta)
    target[:, 1] = np.sin(theta) * np.cos(phi)
    target[:, 2] = np.sin(theta) * np.sin(phi)
    rng = np.random.default_rng(42)
    target[:, 3:] = rng.normal(0.0, 0.01, (n, 509))
    norms = np.linalg.norm(target, axis=1, keepdims=True)
    Y = (target / norms).astype(np.float32)

    train(X, Y, epochs=args.epochs, patience=args.patience, output_path=args.output)
