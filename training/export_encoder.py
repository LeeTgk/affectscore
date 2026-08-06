import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(_HERE, "..", "game", "affectscore", "weights", "encoder_mlp.onnx")


def export_encoder(checkpoint_path, onnx_path=None):
    """Load trained AffectEncoderMLP checkpoint and export to ONNX.

    Args:
        checkpoint_path: Path to .pt state_dict file.
        onnx_path: Output path for encoder_mlp.onnx.
    """
    import torch
    import numpy as np

    if onnx_path is None:
        onnx_path = ONNX_PATH

    sys.path.insert(0, _HERE)
    from train_encoder import _build_model

    model = _build_model(hidden_dim=64, dropout=0.3)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()  # CRITICAL: disables Dropout before ONNX export
    print("[AffectScore] Checkpoint loaded. Dropout disabled (eval mode).")

    dummy = torch.zeros(1, 6, dtype=torch.float32)
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=17,
        input_names=["affect_signal"],
        output_names=["embedding"],
        dynamic_axes={
            "affect_signal": {0: "batch"},
            "embedding": {0: "batch"},
        },
        dynamo=False,  # use legacy TorchScript exporter for compatibility
    )
    print(f"[AffectScore] ONNX exported to {onnx_path}")

    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    assert session.get_inputs()[0].name == "affect_signal", (
        f"Expected input 'affect_signal', got '{session.get_inputs()[0].name}'"
    )
    assert session.get_outputs()[0].name == "embedding", (
        f"Expected output 'embedding', got '{session.get_outputs()[0].name}'"
    )

    dummy_np = np.zeros((1, 6), dtype=np.float32)
    result = session.run(None, {"affect_signal": dummy_np})
    output = result[0]
    assert output.shape == (1, 512), f"Expected shape (1, 512), got {output.shape}"

    l2_norm = float(np.linalg.norm(output[0]))
    assert abs(l2_norm - 1.0) < 1e-5, f"Output not L2-normalized: norm={l2_norm:.6f}"

    print(f"[AffectScore] ONNX round-trip verified: shape={output.shape}, L2-norm={l2_norm:.6f}")
    print("[AffectScore] encoder_mlp.onnx verified (shape, L2-norm, round-trip).")
    return session


def train_from_manifest(manifest_path, epochs=300, patience=30):
    """Train encoder on MERT-auto-labeled manifest and return checkpoint path."""
    import json
    import numpy as np
    sys.path.insert(0, _HERE)
    from train_encoder import train

    with open(manifest_path) as f:
        records = json.load(f)

    print(f"[AffectScore] Training on {len(records)} MERT-annotated clips...")

    # arc_position=0.5 (mid-arc default); engagement signals=0.0 (not in clip-level data)
    X = np.array([
        [r["V_A_valence"], r["V_A_arousal"], 0.5, 0.0, 0.0, 0.0]
        for r in records
    ], dtype=np.float32)

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

    checkpoint_path = os.path.join(_HERE, "encoder_real_corpus.pt")
    model, _ = train(X, Y, epochs=epochs, patience=patience, output_path=checkpoint_path)
    print(f"[AffectScore] Training complete. Checkpoint: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export AffectEncoderMLP to ONNX")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", type=str,
                       help="Path to training_set.json -- train encoder then export")
    group.add_argument("--checkpoint", type=str,
                       help="Path to existing .pt checkpoint -- export only")
    parser.add_argument("--output", type=str, default=ONNX_PATH)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    if args.train:
        checkpoint = train_from_manifest(args.train, args.epochs, args.patience)
    else:
        checkpoint = args.checkpoint

    export_encoder(checkpoint, args.output)
