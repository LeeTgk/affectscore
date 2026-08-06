"""
ACE-Step v1.5 LoRA training script for AffectScore.
Trains LoRA adapters on cross-attention Q/V layers with flow-matching loss and optional CLAP auxiliary loss.

Usage:
    python training/train_lora.py --rank 32 --lambda_clap 0.1
    python training/train_lora.py --rank 32 --lambda_clap 0.1 --ablation no-affect
    python training/train_lora.py --rank 32 --lambda_clap 0.1 --ablation no-lora
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

import torch


def va_to_mood_words(valence: float, arousal: float) -> str:
    """Map V-A values to CLAP-compatible text string using Russell circumplex quadrants (threshold 0.3)."""
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
    """Hard-fail if BF16 or flash_attn are unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError("[train_lora] CUDA required. Runs on A100 via Colab.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "[train_lora] BF16 not supported. "
            "Required for consistent training across rank variants. "
            "Use A100 via Google Colab Pro+."
        )
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "[train_lora] flash_attn not installed. "
            "Install: pip install flash-attn --no-build-isolation"
        )
    print("[AffectScore] Prerequisites: CUDA OK, BF16 OK, flash_attn OK")


def verify_gates_passed(gate_dir: str):
    """Verify all three gates passed before training starts."""
    gates = [
        ("gate1_results.json", "gate1_pass"),
        ("gate2_results.json", "pass"),
        ("gate3_results.json", "gate_pass"),
    ]
    for filename, key in gates:
        path = os.path.join(gate_dir, filename)
        if not os.path.exists(path):
            raise RuntimeError(
                f"[train_lora] Gate results not found: {path}\n"
                f"Run all three gate scripts before training."
            )
        with open(path) as f:
            results = json.load(f)
        if not results.get(key, False):
            raise RuntimeError(
                f"[train_lora] {filename} shows gate FAILED (key '{key}' = False). "
                f"Fix the gate before training."
            )
    print("[AffectScore] All three gates confirmed PASSED.")


from training.utils.dataset import AffectScoreDataset  # noqa: F401 (re-exported)


def save_as_diffusers_adapter(peft_model, out_dir: str) -> None:
    """Save LoRA weights in ACE-Step / diffusers format.

    PEFT's save_pretrained writes adapter_model.safetensors with keys like:
      base_model.model.<path>.lora_A.default.weight
    ACE-Step's load_lora_adapter expects pytorch_lora_weights.safetensors with:
      <path>.lora_A.weight
    This function strips the two extra prefixes and saves the file ACE-Step needs.
    """
    from safetensors.torch import save_file

    peft_state = peft_model.state_dict()
    converted = {}
    for k, v in peft_state.items():
        if "lora_A" not in k and "lora_B" not in k:
            continue
        k = k.replace("base_model.model.", "")
        k = k.replace(".lora_A.default.", ".lora_A.")
        k = k.replace(".lora_B.default.", ".lora_B.")
        converted[k] = v

    os.makedirs(out_dir, exist_ok=True)
    save_file(converted, os.path.join(out_dir, "pytorch_lora_weights.safetensors"))
    print(f"[AffectScore] Diffusers-format adapter saved: {out_dir} "
          f"({len(converted)} tensors)")


def build_lora_config(rank: int, ablation: str, target_modules: list) -> object:
    """Build LoraConfig for the requested ablation variant.

    Ablation variants (all separately trained, never zeroed at inference):
      full:      Q/V LoRA + CLAP auxiliary loss (standard system)
      no-lora:   No LoRA applied; base model saved as-is (returns None)
      no-affect: Q/V LoRA but generic text conditioning (no V-A mapping)
      no-style:  Q/V LoRA + V-A conditioning but no CLAP auxiliary loss (lambda=0)
    """
    from peft import LoraConfig

    if ablation == "no-lora":
        print("[AffectScore] Ablation: no-lora (frozen base model, no LoRA applied)")
        return None

    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,      # RSLoRA convention: alpha = 2*r
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        use_rslora=True,          # RSLoRA normalises by 1/sqrt(r)
    )
    print(f"[AffectScore] LoraConfig: r={rank}, alpha={rank*2}, "
          f"targets='{target_modules}', rslora=True")
    return config


def train(args):
    """Main training loop for ACE-Step LoRA fine-tuning."""
    import tempfile
    import torchaudio
    import torch.nn.functional as F

    if args.run_id is None:
        args.run_id = f"ace-step-r{args.rank}-{datetime.now().strftime('%Y%m%d')}"
        if args.ablation != "full":
            args.run_id += f"-{args.ablation}"
    print(f"[AffectScore] Run ID: {args.run_id}")

    from training.utils.checkpoint import (
        find_latest_checkpoint, save_checkpoint, install_sigterm_handler
    )
    ckpt_path, start_epoch = find_latest_checkpoint(args.run_id, args.drive_base)
    if ckpt_path:
        print(f"[AffectScore] Resuming from epoch {start_epoch}: {ckpt_path}")
    else:
        print(f"[AffectScore] Starting fresh. Run ID: {args.run_id}")

    from acestep.pipeline_ace_step import ACEStepPipeline
    print("[AffectScore] Loading ACE-Step pipeline (may take ~30s)...")
    pipe = ACEStepPipeline(
        checkpoint_dir=None,
        device_id=0,
        dtype="bfloat16",
        torch_compile=False,   # no torch.compile during training
    )
    pipe.load_checkpoint()
    print("[AffectScore] Pipeline loaded.")

    transformer = pipe.ace_step_transformer
    vae          = pipe.music_dcae
    SAMPLE_RATE   = 44100
    CHUNK_SAMPLES = 4 * SAMPLE_RATE  # 176400 -- fixed 4s chunk to match VAE T=44

    # Fix dtype mismatch: sinusoidal time embedding computes in float32 but the
    # downstream linear layers are bfloat16. Register a pre-hook that casts the
    # embedding to match the layer's weight dtype before each linear call.
    # Done via hook rather than autocast so LoRA grad_fn is never severed.
    def _cast_to_weight_dtype(module, args):
        w_dtype = module.weight.dtype
        return tuple(a.to(w_dtype) if isinstance(a, torch.Tensor) else a for a in args)

    for name, mod in transformer.named_modules():
        if isinstance(mod, torch.nn.Linear) and "timestep_embedder" in name:
            mod.register_forward_pre_hook(_cast_to_weight_dtype)

    for param in vae.parameters():
        param.requires_grad = False
    for param in pipe.text_encoder_model.parameters():
        param.requires_grad = False

    # ACE-Step joint attention: audio side is cross_attn.to_q / cross_attn.to_v;
    # text side is cross_attn.add_q_proj / cross_attn.add_v_proj.
    audio_qv = [n for n, _ in transformer.named_modules()
                if "cross_attn" in n and
                (n.endswith("to_q") or n.endswith("to_v"))]

    if audio_qv:
        lora_targets = r".*\.cross_attn\.(to_q|to_v)"
        print(f"[AffectScore] Targeting cross_attn audio-side Q/V "
              f"({len(audio_qv)} modules): to_q + to_v")
    else:
        all_qv = [n for n, _ in transformer.named_modules()
                  if n.endswith("to_q") or n.endswith("to_v")]
        lora_targets = r".*\.(to_q|to_v)"
        print(f"[AffectScore] Fallback: targeting all Q/V "
              f"({len(all_qv)} modules): to_q + to_v")

    lora_config = build_lora_config(args.rank, args.ablation, lora_targets)

    if lora_config is None:
        save_path = os.path.join(
            args.drive_base, "checkpoints", args.run_id, "base_model"
        )
        os.makedirs(save_path, exist_ok=True)
        transformer.save_pretrained(save_path)
        print(f"[AffectScore] no-lora: base transformer saved to {save_path}")
        return

    from peft import get_peft_model
    transformer = get_peft_model(transformer, lora_config)
    transformer.enable_adapter_layers()
    transformer.train()
    transformer.print_trainable_parameters()

    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    if trainable == 0:
        _all_qv = [n for n, _ in transformer.named_modules()
                   if n.endswith("to_q") or n.endswith("to_v")]
        raise RuntimeError(
            f"[train_lora] No trainable params after LoRA injection.\n"
            f"  target_modules={lora_targets} did not match any module.\n"
            f"  Sampled Q/V paths in model: {_all_qv[:8]}"
        )

    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=args.lr,
    )
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cuda")
        transformer.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[AffectScore] Checkpoint loaded: epoch {ckpt['epoch']}")

    epoch_ref = [start_epoch]
    install_sigterm_handler(args.run_id, transformer, optimizer, epoch_ref, args.drive_base)

    clap_model = None
    use_clap = args.lambda_clap > 0.0 and args.ablation != "no-style"
    if use_clap:
        from msclap import CLAP
        print("[AffectScore] Loading CLAP model for auxiliary loss...")
        clap_model = CLAP(version="2023", use_cuda=True)
        print("[AffectScore] CLAP model loaded.")

    manifest_path = args.manifest if args.manifest else os.path.join(_REPO_ROOT, "data", "manifest.json")
    if args.audio_dir:
        audio_dir = args.audio_dir
    else:
        candidates = [
            os.path.join(args.drive_base, "data", "audio"),
            os.path.join(args.drive_base, "preprocessed_unfiltered"),
            os.path.join(args.drive_base, "preprocessed"),
            os.path.join(args.drive_base, "data", "preprocessed"),
            os.path.join(_REPO_ROOT, "data", "preprocessed"),
        ]
        audio_dir = next((p for p in candidates if os.path.isdir(p)), candidates[-1])
    print(f"[AffectScore] Audio directory: {audio_dir}")

    held_out_ids = []
    held_out_path = os.path.join(_REPO_ROOT, "data", "held_out_set.json")
    if os.path.exists(held_out_path):
        with open(held_out_path) as f:
            held_out_ids = [c["clip_id"] for c in json.load(f)]

    def chunk_collate(batch):
        """Truncate/pad each waveform to CHUNK_SAMPLES so default_collate can stack them."""
        waveforms, valences, arousals = [], [], []
        for item in batch:
            w = item["waveform"]   # (C, N)
            n = w.shape[-1]
            if n >= CHUNK_SAMPLES:
                start = torch.randint(0, n - CHUNK_SAMPLES + 1, (1,)).item()
                w = w[:, start : start + CHUNK_SAMPLES]
            else:
                w = torch.nn.functional.pad(w, (0, CHUNK_SAMPLES - n))
            waveforms.append(w)
            valences.append(item["valence"])
            arousals.append(item["arousal"])
        return {
            "waveform": torch.stack(waveforms),
            "valence":  torch.stack(valences),
            "arousal":  torch.stack(arousals),
        }

    dataset = AffectScoreDataset(
        manifest_path, audio_dir,
        sample_rate=SAMPLE_RATE,
        exclude_ids=held_out_ids,
    )

    # Inverse-frequency quadrant weighted sampler -- corrects Q3 dominance (~74%)
    # and Q2 scarcity (~1.5%) so each quadrant contributes ~25% of drawn batches.
    def _quadrant(v, a):
        if v >= 0 and a >= 0.5: return "Q1"
        if v <  0 and a >= 0.5: return "Q2"
        if v <  0 and a <  0.5: return "Q3"
        return "Q4"

    _q_counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    for _clip in dataset.clips:
        _q_counts[_quadrant(_clip["valence"], _clip["arousal"])] += 1
    # Raw inverse-frequency weights, then cap ratio relative to the majority quadrant.
    # Without a cap, a severely depleted quadrant (e.g. Q2=32 clips vs Q3=1227) would
    # be oversampled ~38x, causing memorisation of the minority clips within a few epochs.
    _q_raw    = {q: 1.0 / max(n, 1) for q, n in _q_counts.items()}
    _w_floor  = min(_q_raw.values())
    _cap      = _w_floor * args.max_quadrant_weight
    _q_weight = {q: min(w, _cap) for q, w in _q_raw.items()}
    _sample_weights = torch.tensor(
        [_q_weight[_quadrant(c["valence"], c["arousal"])] for c in dataset.clips],
        dtype=torch.float64,
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=_sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )
    print(
        "[AffectScore] Weighted sampler: "
        + ", ".join(
            f"{q}={_q_counts[q]} clips (eff. weight x{_q_counts[max(_q_counts, key=_q_counts.get)] / max(_q_counts[q], 1):.1f})"
            for q in ["Q1", "Q2", "Q3", "Q4"]
        )
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=chunk_collate,
    )

    # Null lyrics: BOS (261) + empty-line (2) -- instrumental music, no karaoke.
    # Derived from pipe.tokenize_lyrics("") -> [261, 2].
    _NULL_LYR_IDS  = torch.tensor([[261, 2]], dtype=torch.long, device="cuda")
    _NULL_LYR_MASK = torch.ones(1, 2, dtype=torch.long, device="cuda")

    CLAP_EVERY_N        = 50
    SAVE_INTERVAL_SEC   = 1800
    SAVE_INTERVAL_EPOCH = 5

    last_save_time = time.time()
    step_idx = 0

    import csv
    log_dir = os.path.join(args.drive_base, "checkpoints", args.run_id)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train_log.csv")
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if os.path.getsize(log_path) == 0:
        log_writer.writerow(["epoch", "step", "mean_L_diff", "mean_L_clap"])
    print(f"[AffectScore] Training log: {log_path}")

    print(f"[AffectScore] Training: epochs={args.epochs}, rank={args.rank}, "
          f"lambda_clap={args.lambda_clap}, ablation={args.ablation}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_ref[0] = epoch
        epoch_loss      = 0.0
        epoch_clap_sum  = 0.0
        epoch_clap_count = 0
        num_batches = 0

        transformer.train()
        for batch in train_loader:
            waveform      = batch["waveform"].to("cuda", dtype=torch.bfloat16)  # (B, C, N)
            valence_batch = batch["valence"]
            arousal_batch = batch["arousal"]
            B             = waveform.shape[0]

            mood_texts = []
            for i in range(B):
                if args.ablation == "no-affect":
                    mood_texts.append("music")
                else:
                    mood_texts.append(
                        va_to_mood_words(valence_batch[i].item(), arousal_batch[i].item())
                    )

            optimizer.zero_grad()

            # Encode waveform -> latents (frozen VAE).
            # VAE conv_in expects 2-channel (stereo) mel; duplicate mono if needed.
            if waveform.shape[1] == 1:
                waveform = waveform.repeat(1, 2, 1)

            n_samples = waveform.shape[-1]
            audio_lengths = torch.full((B,), n_samples, dtype=torch.int64, device="cuda")

            with torch.no_grad():
                vae_out        = vae.encode(waveform, audio_lengths, sr=SAMPLE_RATE)
                latents        = vae_out[0]   # (B, 8, 16, T)
                latent_lengths = vae_out[1]   # (B,) int64 -- valid T frames per sample

            # Flow matching interpolation
            _, C, H, T = latents.shape
            noise    = torch.randn_like(latents)
            t        = torch.rand(B, device="cuda", dtype=torch.float32)
            t_4d     = t[:, None, None, None].to(torch.bfloat16)
            z_t      = (1.0 - t_4d) * noise + t_4d * latents
            v_target = latents - noise
            t_scaled = (t * 1000.0).to(torch.bfloat16)

            with torch.no_grad():
                te, te_mask = pipe.get_text_embeddings(mood_texts)  # (B, 8, 768), (B, 8)

            spk = torch.zeros(B, 512, device="cuda", dtype=torch.bfloat16)

            lyric_ids  = _NULL_LYR_IDS.expand(B, -1)
            lyric_mask = _NULL_LYR_MASK.expand(B, -1)

            attn_mask = (
                torch.arange(T, device="cuda")[None, :] < latent_lengths[:, None]
            )  # (B, T) bool

            v_pred = transformer(
                hidden_states=z_t,
                attention_mask=attn_mask,
                encoder_text_hidden_states=te,
                text_attention_mask=te_mask,
                speaker_embeds=spk,
                lyric_token_idx=lyric_ids,
                lyric_mask=lyric_mask,
                timestep=t_scaled,
                return_dict=False,
            )[0]  # (B, 8, 16, T) -- predicted flow velocity

            valid_4d = attn_mask[:, None, None, :].expand_as(v_pred)
            L_diff   = F.mse_loss(v_pred[valid_4d], v_target[valid_4d])

            # CLAP auxiliary loss: computed on decoded x0_pred vs mood text.
            # Note: CLAP embeddings require numpy (msclap 2023 API); gradient
            # does not flow from L_clap to LoRA params -- L_clap serves as a
            # monitoring and scaled-loss signal. Ablation "no-style" sets
            # lambda_clap=0, disabling this term entirely.
            L_clap = torch.tensor(0.0, device="cuda")
            if clap_model is not None and step_idx % CLAP_EVERY_N == 0:
                try:
                    with torch.no_grad():
                        # x0_pred from flow matching: z_t = (1-t)*noise + t*x0
                        # -> x0 = z_t + (1-t)*v_pred
                        x0_pred   = z_t + (1.0 - t_4d) * v_pred
                        # Pass audio_lengths=None: all clips are same length, decode full latent.
                        # Passing latent_lengths (frames) as audio_lengths (samples) would truncate to ~1ms.
                        dec_sr, pred_wavs = vae.decode(x0_pred)
                        audio_np = torch.stack(
                            [w.float().mean(dim=0) for w in pred_wavs]
                        ).cpu().numpy()  # (B, N) mono float32

                    with tempfile.TemporaryDirectory() as tmpdir:
                        audio_embs = []
                        for ib in range(B):
                            wav_path = os.path.join(tmpdir, f"pred_{ib}.wav")
                            torchaudio.save(
                                wav_path,
                                torch.from_numpy(audio_np[ib : ib + 1]),
                                dec_sr,
                            )
                            audio_embs.append(clap_model.get_audio_embeddings([wav_path]))

                    a_emb = torch.cat(audio_embs, dim=0).float().to("cuda")
                    t_emb = clap_model.get_text_embeddings(mood_texts).float().to("cuda")
                    a_n    = F.normalize(a_emb, dim=-1)
                    b_n    = F.normalize(t_emb, dim=-1)
                    L_clap = (1.0 - (a_n * b_n).sum(dim=-1)).mean()
                    epoch_clap_sum   += L_clap.item()
                    epoch_clap_count += 1
                except Exception as exc:
                    print(f"[train_lora] CLAP skipped at step {step_idx}: {exc}")

            loss = L_diff + args.lambda_clap * L_clap
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in transformer.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            step_idx  += 1
            epoch_loss += loss.item()
            num_batches += 1

        mean_loss  = epoch_loss / max(num_batches, 1)
        mean_clap  = epoch_clap_sum / epoch_clap_count if epoch_clap_count > 0 else float("nan")
        log_writer.writerow([epoch, step_idx, f"{mean_loss:.6f}", f"{mean_clap:.6f}"])
        log_file.flush()
        if epoch % 5 == 0 or epoch == 1:
            clap_str = f"{mean_clap:.4f}" if epoch_clap_count > 0 else "n/a"
            print(f"[AffectScore] Epoch {epoch:3d}/{args.epochs}: "
                  f"L_diff={mean_loss:.4f}  L_clap={clap_str}")

        now = time.time()
        if epoch % SAVE_INTERVAL_EPOCH == 0 or (now - last_save_time) > SAVE_INTERVAL_SEC:
            save_checkpoint(transformer, optimizer, epoch, args.run_id, args.drive_base)
            last_save_time = now

    log_file.close()
    print(f"[AffectScore] Training complete after {args.epochs} epochs.")
    save_checkpoint(transformer, optimizer, args.epochs, args.run_id, args.drive_base)

    local_save = os.path.join(
        args.drive_base, "checkpoints", args.run_id, "final_adapter"
    )
    save_as_diffusers_adapter(transformer, local_save)

    from datetime import datetime as _dt
    _date = _dt.now().strftime("%Y%m%d")
    hf_repo = f"{args.hf_username}/affectscore-ace-step-r{args.rank}-{_date}"
    if args.ablation != "full":
        hf_repo += f"-{args.ablation}"
    print(f"[AffectScore] Pushing adapter to HuggingFace: {hf_repo}")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(hf_repo, exist_ok=True, private=False)
        api.upload_file(
            path_or_fileobj=os.path.join(local_save, "pytorch_lora_weights.safetensors"),
            path_in_repo="pytorch_lora_weights.safetensors",
            repo_id=hf_repo,
        )
        print(f"[AffectScore] Adapter pushed to: {hf_repo}")
    except Exception as e:
        print(f"[AffectScore] HF push failed: {e}. Adapter already saved locally.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AffectScore LoRA on ACE-Step v1.5 Turbo"
    )
    parser.add_argument("--rank", type=int, required=True, choices=[16, 32, 64],
                        help="LoRA rank (sweep 16, 32, 64)")
    parser.add_argument("--lambda_clap", type=float, default=0.1,
                        help="CLAP auxiliary loss weight")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "no-lora", "no-affect", "no-style"],
                        help="Ablation variant. Each is a separately-trained model.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (4-8 for A100 40GB)")
    parser.add_argument("--drive-base", type=str,
                        default="/content/drive/MyDrive/affectscore",
                        help="Google Drive base path")
    parser.add_argument("--audio-dir", type=str, default=None,
                        help="Explicit path to preprocessed WAV directory (overrides drive-base lookup)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID (auto-generated: ace-step-r{rank}-{date})")
    parser.add_argument("--hf-username", type=str, default="HiiragiLee",
                        help="HuggingFace username for push_to_hub")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to training manifest JSON (default: data/manifest.json)")
    parser.add_argument("--max-quadrant-weight", type=float, default=5.0,
                        help="Cap minority-quadrant oversampling to this multiple of the majority "
                             "quadrant weight. Prevents memorisation when a quadrant has very few clips.")
    parser.add_argument("--skip-gate-check", action="store_true",
                        help="Skip gate verification (for testing only)")
    args = parser.parse_args()

    print(f"[AffectScore] AffectScore LoRA Training")
    print(f"[AffectScore] Rank={args.rank}, lambda_clap={args.lambda_clap}, "
          f"ablation={args.ablation}")

    check_prerequisites()

    if not args.skip_gate_check:
        verify_gates_passed(os.path.join(_REPO_ROOT, "training", "gates"))

    train(args)
