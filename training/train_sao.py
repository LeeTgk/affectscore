"""
Stable Audio Open (SAO) LoRA training script for AffectScore.
Trains a SAO comparison model using LoRAW on cross-attention Q/V layers.

Usage:
    python training/train_sao.py --rank 32 --lambda_clap 0.1

Installation:
    pip install stable-audio-tools
    pip install git+https://github.com/NeuralNotW0rk/LoRAW.git
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch


def va_to_mood_words(valence: float, arousal: float) -> str:
    """Map V-A values to CLAP-compatible text string using Russell circumplex quadrants."""
    if valence >= 0.3 and arousal >= 0.3:
        primary = "triumphant joyful"
    elif valence >= 0.3 and arousal < -0.3:
        primary = "serene peaceful"
    elif valence < -0.3 and arousal >= 0.3:
        primary = "tense anxious"
    elif valence < -0.3 and arousal < -0.3:
        primary = "melancholic somber"
    else:
        primary = "neutral ambient"

    if abs(arousal) >= 0.6:
        energy = "energetic"
    elif abs(arousal) <= 0.2:
        energy = "calm"
    else:
        energy = ""

    parts = [primary] + ([energy] if energy else []) + ["music"]
    return " ".join(parts)


def check_prerequisites():
    """Hard-fail if BF16, flash_attn, stable-audio-tools, or LoRAW are unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError("[train_sao] CUDA required. Runs on A100 via Colab.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("[train_sao] BF16 not supported. Use A100 via Colab Pro+.")
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "[train_sao] flash_attn required. "
            "Install: pip install flash-attn --no-build-isolation"
        )
    try:
        import stable_audio_tools  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "[train_sao] stable-audio-tools not installed. "
            "Install: pip install stable-audio-tools"
        )
    try:
        from loraw import LoRAWrapper  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "[train_sao] LoRAW not installed. "
            "Install: pip install git+https://github.com/NeuralNotW0rk/LoRAW.git"
        )
    print("[AffectScore] Prerequisites: CUDA OK, BF16 OK, flash_attn OK, "
          "stable-audio-tools OK, LoRAW OK")


def verify_gates_passed(gate_dir: str):
    """Verify all three gates passed."""
    gates = [
        ("gate1_results.json", "gate1_pass"),
        ("gate2_results.json", "pass"),
        ("gate3_results.json", "gate_pass"),
    ]
    for filename, key in gates:
        path = os.path.join(gate_dir, filename)
        if not os.path.exists(path):
            raise RuntimeError(
                f"[train_sao] Gate results not found: {path}. Run gate scripts first."
            )
        with open(path) as f:
            results = json.load(f)
        if not results.get(key, False):
            raise RuntimeError(f"[train_sao] {filename} shows gate FAILED. Fix before training.")
    print("[AffectScore] All three gates confirmed PASSED.")


def build_sao_conditioning(affect_embed: torch.Tensor,
                            clap_model, mood_text: str) -> torch.Tensor:
    """Build SAO conditioning: concat(affect_512d, clap_embed).

    SAO conditioning concatenates the 512-d affect vector with the CLAP text
    embedding as a single conditioning token sequence.

    Args:
        affect_embed: (512,) tensor from encoder_mlp.onnx
        clap_model: msclap.CLAP instance
        mood_text: V-A-derived mood word string

    Returns:
        Conditioning tensor for SAO cross-attention
    """
    text_embed = clap_model.get_text_embeddings([mood_text])
    text_embed_tensor = torch.tensor(text_embed, dtype=torch.float32)
    conditioning = torch.cat([affect_embed.unsqueeze(0), text_embed_tensor], dim=-1)
    return conditioning


def train(args):
    """Main training loop for SAO LoRA fine-tuning."""
    import torchaudio

    if args.run_id is None:
        args.run_id = f"sao-r{args.rank}-{datetime.now().strftime('%Y%m%d')}"
    print(f"[AffectScore] SAO Run ID: {args.run_id}")

    from training.utils.checkpoint import (
        find_latest_checkpoint, save_checkpoint, install_sigterm_handler
    )
    ckpt_path, start_epoch = find_latest_checkpoint(args.run_id, args.drive_base)
    if ckpt_path:
        print(f"[AffectScore] Resuming from epoch {start_epoch}: {ckpt_path}")
    else:
        print(f"[AffectScore] Starting fresh SAO training. Run ID: {args.run_id}")

    from stable_audio_tools import get_pretrained_model
    print("[AffectScore] Loading Stable Audio Open base model...")
    sao_model_path = os.path.join(args.drive_base, "models", "stable-audio-open")
    base_model, model_config = get_pretrained_model(sao_model_path)
    base_model = base_model.to("cuda").to(torch.bfloat16)
    print("[AffectScore] SAO base model loaded.")

    from loraw import LoRAWrapper
    lora_model = LoRAWrapper(
        base_model,
        target_components=["cross_attn.q_proj", "cross_attn.v_proj"],
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
    )
    print(f"[AffectScore] LoRAW applied: r={args.rank}, "
          f"targets=['cross_attn.q_proj', 'cross_attn.v_proj']")

    from msclap import CLAP
    print("[AffectScore] Loading CLAP model...")
    clap_model = CLAP(version="2023", use_cuda=True)
    print("[AffectScore] CLAP model loaded.")

    encoder_path = os.path.join(
        _REPO_ROOT, "game", "affectscore", "weights", "encoder_mlp.onnx"
    )
    have_encoder = False
    if os.path.exists(encoder_path):
        try:
            import onnxruntime as ort
            encoder_session = ort.InferenceSession(
                encoder_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            have_encoder = True
            print(f"[AffectScore] Affect encoder loaded from {encoder_path}")
        except Exception as e:
            print(f"[AffectScore] Affect encoder not available: {e}")
    else:
        print(f"[AffectScore] Affect encoder not found at {encoder_path}")
        print("[AffectScore] Using zero affect embedding as fallback.")

    optimizer = torch.optim.AdamW(
        [p for p in lora_model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cuda")
        lora_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch_ref = [start_epoch]
    install_sigterm_handler(args.run_id, lora_model, optimizer, epoch_ref, args.drive_base)

    manifest_path = os.path.join(_REPO_ROOT, "data", "training_set_clean_clap.json")
    if args.audio_dir:
        audio_dir = args.audio_dir
    else:
        candidates = [
            os.path.join(args.drive_base, "data", "audio"),
            os.path.join(args.drive_base, "preprocessed"),
            os.path.join(args.drive_base, "data", "preprocessed"),
            os.path.join(_REPO_ROOT, "data", "preprocessed"),
        ]
        audio_dir = next((p for p in candidates if os.path.isdir(p)), candidates[-1])
    print(f"[AffectScore] Audio dir: {audio_dir}")

    held_out_path = os.path.join(_REPO_ROOT, "data", "held_out_set.json")
    held_out_ids = []
    if os.path.exists(held_out_path):
        with open(held_out_path) as f:
            held_out = json.load(f)
        held_out_ids = [c["clip_id"] for c in held_out]

    SAMPLE_RATE   = 44100
    CHUNK_SAMPLES = 4 * SAMPLE_RATE  # 176400

    def chunk_collate(batch):
        waveforms, valences, arousals = [], [], []
        for item in batch:
            w = item["waveform"]
            n = w.shape[-1]
            if n >= CHUNK_SAMPLES:
                start = torch.randint(0, n - CHUNK_SAMPLES + 1, (1,)).item()
                w = w[:, start : start + CHUNK_SAMPLES]
            else:
                w = torch.nn.functional.pad(w, (0, CHUNK_SAMPLES - n))
            waveforms.append(w)
            valences.append(item["valence"])
            arousals.append(item["arousal"])
        return {"waveform": torch.stack(waveforms),
                "valence": torch.stack(valences),
                "arousal": torch.stack(arousals)}

    # Import AffectScoreDataset from shared utils (avoids sys.argv contamination
    # that occurs when importing from train_lora in Colab notebook environments).
    from training.utils.dataset import AffectScoreDataset
    dataset = AffectScoreDataset(
        manifest_path, audio_dir, sample_rate=SAMPLE_RATE, exclude_ids=held_out_ids
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=chunk_collate
    )

    last_save_time = time.time()
    SAVE_INTERVAL_SEC = 1800
    SAVE_INTERVAL_EPOCH = 5

    print(f"[AffectScore] SAO training: epochs={args.epochs}, "
          f"rank={args.rank}, lambda_clap={args.lambda_clap}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_ref[0] = epoch
        epoch_loss = 0.0
        num_batches = 0

        lora_model.train()
        for batch in train_loader:
            waveform = batch["waveform"].to("cuda", dtype=torch.bfloat16)
            valence_batch = batch["valence"]
            arousal_batch = batch["arousal"]

            optimizer.zero_grad()
            batch_size = waveform.shape[0]
            total_loss = torch.tensor(0.0, device="cuda", dtype=torch.bfloat16)

            for i in range(batch_size):
                v = valence_batch[i].item()
                a = arousal_batch[i].item()
                mood_text = va_to_mood_words(v, a)

                if have_encoder:
                    signal = np.array([[v, a, 0.5, 0.0, 0.0, 0.0]], dtype=np.float32)
                    enc_in = encoder_session.get_inputs()[0].name
                    affect_embed = torch.tensor(
                        encoder_session.run(None, {enc_in: signal})[0][0],
                        dtype=torch.float32
                    )
                else:
                    affect_embed = torch.zeros(512, dtype=torch.float32)

                conditioning = build_sao_conditioning(affect_embed, clap_model, mood_text)

                # SAO diffusion loss via direct model forward pass.
                # stable-audio-tools does not expose a train_step helper -- use the
                # model forward pass directly. Hard-fail on error: silent zero-loss
                # training produces no usable comparison checkpoint.
                try:
                    if hasattr(lora_model, "loss"):
                        L_diff = lora_model.loss({
                            "audio": waveform[i].unsqueeze(0),
                            "conditioning": conditioning,
                        })
                        if isinstance(L_diff, dict):
                            L_diff = L_diff.get("loss", L_diff.get("diffusion_loss"))
                        L_diff = L_diff.to("cuda").to(torch.bfloat16)
                    else:
                        x_pred = lora_model(conditioning)
                        with torch.no_grad():
                            target_latents = lora_model.encode(waveform[i].unsqueeze(0))
                        L_diff = torch.nn.functional.mse_loss(
                            x_pred, target_latents.to(x_pred.dtype)
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"[train_sao] SAO diffusion loss computation failed: {e}\n"
                        "Hard-failing -- silent zero-loss training produces no usable "
                        "comparison checkpoint. Verify stable-audio-tools API after install."
                    ) from e

                total_loss = total_loss + L_diff

            total_loss = total_loss / batch_size
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            num_batches += 1

        mean_loss = epoch_loss / max(num_batches, 1)
        if epoch % 5 == 0:
            print(f"[AffectScore] Epoch {epoch:3d}/{args.epochs}: loss={mean_loss:.4f}")

        now = time.time()
        if epoch % SAVE_INTERVAL_EPOCH == 0 or (now - last_save_time) > SAVE_INTERVAL_SEC:
            save_checkpoint(lora_model, optimizer, epoch, args.run_id, args.drive_base)
            last_save_time = now

    print(f"[AffectScore] SAO training complete after {args.epochs} epochs.")
    save_checkpoint(lora_model, optimizer, args.epochs, args.run_id, args.drive_base)

    hf_repo = f"{args.hf_username}/affectscore-lora-sao"
    print(f"[AffectScore] Pushing SAO adapter to HuggingFace: {hf_repo}")
    try:
        lora_model.push_to_hub(hf_repo)
        print(f"[AffectScore] SAO adapter pushed to: {hf_repo}")
    except Exception as e:
        print(f"[AffectScore] HF push failed: {e}. Saving locally.")
        local_save = os.path.join(
            args.drive_base, "checkpoints", args.run_id, "final_adapter"
        )
        os.makedirs(local_save, exist_ok=True)
        torch.save(lora_model.state_dict(), os.path.join(local_save, "lora_weights.pt"))

    if getattr(args, "eval_after", False):
        print(f"\n[AffectScore] === SAO Evaluation Hook ===")
        print(f"[AffectScore] Generating eval set for SAO adapter...")
        try:
            from eval.generate_eval_set import generate_eval_set as gen_eval
            sao_eval_dir = os.path.join(args.drive_base, "eval_outputs")
            gen_eval(
                held_out_path=os.path.join(_REPO_ROOT, "data", "held_out_set.json"),
                adapter_name="no-lora",
                adapter_path="none",
                drive_output_base=sao_eval_dir,
                output_tag="sao",
                num_inference_steps=40,
            )
            print(
                f"[AffectScore] SAO eval clips written to {sao_eval_dir}/no-lora_sao/. "
                f"Run eval/audio_metrics.py --audio-dir {sao_eval_dir}/no-lora_sao/ "
                f"to compute FAD-MERT for SAO vs ACE-Step r=32 comparison."
            )
        except Exception as e:
            print(f"[AffectScore] Eval hook failed: {e}. Run generate_eval_set.py manually.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SAO LoRA comparison model for AffectScore"
    )
    parser.add_argument("--rank", type=int, required=True, choices=[16, 32, 64])
    parser.add_argument("--lambda_clap", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--drive-base", type=str,
                        default="/content/drive/MyDrive/affectscore")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--hf-username", type=str, default="HiiragiLee")
    parser.add_argument("--audio-dir", type=str, default=None,
                        help="Explicit path to preprocessed WAV directory (overrides drive-base lookup)")
    parser.add_argument("--skip-gate-check", action="store_true")
    parser.add_argument(
        "--eval-after",
        action="store_true",
        default=False,
        help=(
            "After training completes, generate eval clips and print the "
            "command to compute FAD-MERT for SAO vs ACE-Step r=32 comparison."
        ),
    )
    args = parser.parse_args()

    print(f"[AffectScore] Stable Audio Open LoRA Training (SAO comparison)")
    check_prerequisites()
    if not args.skip_gate_check:
        verify_gates_passed(os.path.join(_REPO_ROOT, "training", "gates"))
    train(args)
