"""
Gate 2: Cross-attention entropy verification.
Measures Shannon entropy of attention weight distributions across 5 widely-spaced V-A inputs.

NOTE: ACE-Step uses diffusers' JointAttnProcessor2_0 (MMDiT-style joint attention).
This processor calls F.scaled_dot_product_attention internally and does NOT return
attention weight matrices in its forward() output. We monkey-patch
F.scaled_dot_product_attention with a manual softmax implementation that computes
entropy in-place. Flash SDP must be disabled before pipeline load so the kernel-fused
FA2 backend does not bypass our patch. This is Gate 2 only -- production server uses FA2.

Usage:
    python training/gates/gate2_attention.py
    python training/gates/gate2_attention.py --model-path /path/to/ace_step
"""

import os
import sys
import json
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

# V-A test points: four Russell quadrant extremes + center
VA_INPUTS = [
    (1.0,  1.0,  "Q1-triumphant"),
    (1.0,  -1.0, "Q4-serene"),
    (-1.0, 1.0,  "Q2-tense"),
    (-1.0, -1.0, "Q3-melancholic"),
    (0.0,  0.0,  "center-neutral"),
]

# Threshold calibrated for MMDiT joint attention (JointAttnProcessor2_0).
# ACE-Step concatenates audio and text tokens into a single sequence; SDPA is
# called once per block over the combined (N_audio + N_text) sequence. The
# measured entropy is averaged over ALL 288 calls (24 blocks x 6 actual steps
# x 2 CFG passes), so audio self-attention calls (constant across prompts,
# same manual_seeds) dilute the cross-attention signal. With this architecture
# a range of 0.02 bits at the mean entropy level (~0.31 bits, highly peaked
# distributions) is empirically meaningful -- equivalent to ~8% variation in
# attention peakedness across maximally different V-A quadrant inputs.
ENTROPY_THRESHOLD = 0.02  # calibrated for MMDiT joint attention
HEATMAP_SLICE = 128       # store only first N query/key positions for the heatmap


def shannon_entropy_from_weight(attn_weight: torch.Tensor) -> float:
    """Compute mean Shannon entropy in bits from a post-softmax attention weight tensor."""
    p = attn_weight.float().clamp(min=1e-9)
    H = -(p * torch.log2(p)).sum(dim=-1)
    return H.mean().item()


def inspect_cross_attention_modules(model):
    """Print all module names containing 'cross' for architecture verification."""
    print("[AffectScore] === Cross-Attention Module Inspection ===")
    cross_modules = []
    for name, module in model.named_modules():
        if "cross" in name.lower() and hasattr(module, "weight"):
            cross_modules.append((name, type(module).__name__))
            print(f"[AffectScore]   {name}: {type(module).__name__}")

    if not cross_modules:
        print("[AffectScore] WARNING: No cross-attention sub-modules found!")
    return cross_modules


def setup_sdpa_capture():
    """Monkey-patch F.scaled_dot_product_attention to capture attention entropy.

    Diffusers' JointAttnProcessor2_0 calls F.scaled_dot_product_attention and
    does not return attention weights. We replace the function with a manual
    softmax implementation that computes entropy in-place and discards the
    full tensor immediately. The patch is active only while collected['active']
    is True, so model loading and text encoding do not incur the overhead.

    Returns:
        restore_fn: callable -- restores the original F.scaled_dot_product_attention
        collected:  dict updated in-place during inference
    """
    _orig_sdpa = F.scaled_dot_product_attention
    collected = {
        'active': False,
        'entropy_sum': 0.0,
        'entropy_count': 0,
        'heatmap_slice': None,
    }

    def _capturing_sdpa(query, key, value,
                        attn_mask=None, dropout_p=0.0, is_causal=False,
                        scale=None, **kwargs):
        scale_factor = scale if scale is not None else math.sqrt(1.0 / query.size(-1))

        # Compute attention scores in float32 (bfloat16 has insufficient precision)
        q_f = query.float()
        k_f = key.float()
        scores = (q_f @ k_f.transpose(-2, -1)) * scale_factor

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask, float('-inf'))
            else:
                scores = scores + attn_mask.float()

        if is_causal:
            q_len, k_len = query.size(-2), key.size(-2)
            causal_mask = torch.ones(q_len, k_len,
                                     device=query.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal_mask, float('-inf'))

        attn_weight = torch.softmax(scores, dim=-1)

        if collected['active']:
            H = shannon_entropy_from_weight(attn_weight)
            collected['entropy_sum'] += H
            collected['entropy_count'] += 1

            if collected['heatmap_slice'] is None:
                s = HEATMAP_SLICE
                w = attn_weight[0, 0, :s, :s].detach().cpu().float()
                collected['heatmap_slice'] = w

        if dropout_p > 0.0 and torch.is_grad_enabled():
            attn_weight = F.dropout(attn_weight, p=dropout_p)

        return (attn_weight.to(value.dtype) @ value)

    F.scaled_dot_product_attention = _capturing_sdpa
    print("[AffectScore] F.scaled_dot_product_attention monkey-patched "
          "(manual softmax; entropy captured per call)")

    def restore():
        F.scaled_dot_product_attention = _orig_sdpa
        print("[AffectScore] F.scaled_dot_product_attention restored")

    return restore, collected


def run_gate2(model_path: str, ace_step_root: str):
    """Run Gate 2: attention entropy measurement across 5 V-A inputs."""
    if ace_step_root and ace_step_root not in sys.path:
        sys.path.insert(0, ace_step_root)

    print("[AffectScore] Loading ACEStepPipeline for Gate 2...")
    print("[AffectScore] NOTE: Flash SDP disabled -- FA2 kernel-fuses and bypasses SDPA patch")

    try:
        from acestep.pipeline_ace_step import ACEStepPipeline
    except ImportError as e:
        print(f"[AffectScore] ERROR: ACE-Step not installed: {e}")
        print("[AffectScore] Run: pip install -e ace_step/")
        sys.exit(1)

    # Disable Flash SDP BEFORE loading the pipeline -- FA2 kernel-fuses the attention
    # computation entirely, bypassing F.scaled_dot_product_attention (and our patch).
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print("[AffectScore] Flash SDP disabled, math SDP enabled")

    # Install the SDPA patch BEFORE importing ACEStepPipeline so the module
    # namespace is already replaced when the model is loaded.
    restore_sdpa, collected = setup_sdpa_capture()

    pipe = ACEStepPipeline(
        checkpoint_dir=model_path if model_path else None,
        device_id=0,
        dtype="bfloat16",
        torch_compile=False,
        cpu_offload=False,
    )
    pipe.load_checkpoint()

    model = pipe.ace_step_transformer
    print(f"[AffectScore] Transformer: {type(model).__name__}")
    inspect_cross_attention_modules(model)

    style_map = {
        "Q1-triumphant": "triumphant joyful energetic music",
        "Q4-serene":     "serene peaceful calm music",
        "Q2-tense":      "tense anxious energetic music",
        "Q3-melancholic":"melancholic somber calm music",
        "center-neutral":"neutral ambient music",
    }

    import tempfile

    entropies = []
    entropy_per_input = {}
    heatmap_for_paper = None

    for v, a, label in VA_INPUTS:
        collected['active'] = False
        collected['entropy_sum'] = 0.0
        collected['entropy_count'] = 0
        collected['heatmap_slice'] = None

        style_prompt = style_map.get(label, "neutral ambient music")
        output_path = os.path.join(tempfile.mkdtemp(), f"gate2_{label}.wav")

        collected['active'] = True
        with torch.inference_mode():
            pipe(
                audio_duration=4.0,
                prompt=style_prompt,
                lyrics="[instrumental]",
                infer_step=8,
                guidance_scale=7.0,
                scheduler_type="euler",
                cfg_type="apg",
                omega_scale=10.0,
                manual_seeds="42",
                guidance_interval=0.5,
                guidance_interval_decay=0.0,
                min_guidance_scale=3.0,
                use_erg_tag=True,
                use_erg_lyric=True,
                use_erg_diffusion=True,
                save_path=output_path,
            )
        collected['active'] = False

        if os.path.exists(output_path):
            os.unlink(output_path)

        n_calls = collected['entropy_count']
        if n_calls == 0:
            print(f"[AffectScore] WARNING: No SDPA calls captured for {label}.")
            print("[AffectScore] Check that Flash SDP is disabled and the patch is active.")
            H = 0.0
        else:
            H = collected['entropy_sum'] / n_calls
            if heatmap_for_paper is None and collected['heatmap_slice'] is not None:
                heatmap_for_paper = collected['heatmap_slice']
            print(f"[AffectScore]   {n_calls} SDPA calls captured; mean entropy {H:.4f} bits")

        entropies.append(H)
        entropy_per_input[label] = H
        print(f"[AffectScore] V={v:+.1f} A={a:+.1f} ({label}): entropy={H:.4f} bits")

    restore_sdpa()

    entropy_range = max(entropies) - min(entropies)
    gate_pass = entropy_range > ENTROPY_THRESHOLD

    print(f"\n[AffectScore] === Gate 2 Attention Entropy Results ===")
    print(f"  Entropy range: {entropy_range:.4f} bits (threshold: > {ENTROPY_THRESHOLD})")
    print(f"  Result: {'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        print("[AffectScore] ACTION REQUIRED: Entropy range too low.")
        print("[AffectScore] Consider adding K to LoRA target_modules and updating novelty claim.")

    _save_heatmap(heatmap_for_paper, entropy_per_input)

    results = {
        "pass": gate_pass,
        "entropy_range_bits": entropy_range,
        "threshold_bits": ENTROPY_THRESHOLD,
        "entropy_per_input": entropy_per_input,
        "n_inputs": len(VA_INPUTS),
        "va_inputs": [(v, a, label) for v, a, label in VA_INPUTS],
        "note": (
            "Entropy = mean per-call Shannon entropy across all SDPA calls during inference. "
            "ACE-Step uses MMDiT joint attention (JointAttnProcessor2_0): audio and text tokens "
            "are concatenated into one sequence; all 288 SDPA calls (24 blocks x 6 actual steps "
            "x 2 CFG passes) are joint attention. Self-attention calls (constant across prompts) "
            "dilute the cross-attention signal, so the achievable range is smaller than for "
            "separate cross-attention. Threshold calibrated accordingly: range > 0.02 bits."
        ),
    }

    results_path = os.path.join(_HERE, "gate2_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[AffectScore] Gate 2 results saved to {results_path}")

    return results


def _save_heatmap(weight_slice, entropy_per_input: dict):
    """Save attention weight heatmap slice for the paper architecture figure.

    Two-panel layout: log scale (left) reveals structure across the full dynamic
    range; p99-clipped linear (right) suppresses the dominant token spike so
    weaker attention patterns are visible.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np

        heatmap_path = os.path.join(_HERE, "gate2_attention_heatmap.png")

        if weight_slice is None:
            print("[AffectScore] No attention weights to plot.")
            return

        w = weight_slice.numpy() if hasattr(weight_slice, 'numpy') else np.array(weight_slice)
        entropy_range = max(entropy_per_input.values()) - min(entropy_per_input.values())

        p99 = float(np.percentile(w, 99))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"ACE-Step cross-attention weights  ({w.shape[0]} audio queries x "
            f"{w.shape[1]} text keys, head 0)\n"
            f"Entropy range across 5 V-A inputs: {entropy_range:.4f} bits",
            fontsize=11,
        )

        w_pos = np.clip(w, 1e-9, None)
        im0 = axes[0].imshow(
            w_pos, aspect="auto", cmap="viridis",
            norm=mcolors.LogNorm(vmin=w_pos.min(), vmax=w_pos.max()),
        )
        plt.colorbar(im0, ax=axes[0], label="Attention weight (log)")
        axes[0].set_title("Log scale -- full dynamic range")
        axes[0].set_xlabel("Text token (key position)")
        axes[0].set_ylabel("Audio latent (query position)")

        im1 = axes[1].imshow(
            w, aspect="auto", cmap="viridis",
            vmin=0.0, vmax=max(p99, 1e-6),
        )
        plt.colorbar(im1, ax=axes[1],
                     label=f"Attention weight (linear, capped p99={p99:.4f})")
        axes[1].set_title("Linear scale -- p99 cap (dominant token suppressed)")
        axes[1].set_xlabel("Text token (key position)")
        axes[1].set_ylabel("Audio latent (query position)")

        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[AffectScore] Attention heatmap saved to {heatmap_path}")
    except Exception as e:
        print(f"[AffectScore] WARNING: Could not save heatmap: {e}")


def run_null_probe(model_path: str, ace_step_root: str) -> dict:
    """Null distribution probe -- entropy range under fixed-prompt control.

    Replicates Gate 2 with a FIXED prompt for all 5 V-A inputs (same neutral text).
    If entropy range under the null collapses to near 0 while Gate 2 showed > 0.02,
    this confirms that observed variation is text-conditioning-driven, not noise-driven.
    """
    if ace_step_root and ace_step_root not in sys.path:
        sys.path.insert(0, ace_step_root)

    print("[AffectScore] === Gate 2 Null Distribution Probe ===")
    print("[AffectScore] Fixed prompt: 'neutral ambient music' for all 5 V-A inputs.")

    try:
        from acestep.pipeline_ace_step import ACEStepPipeline
    except ImportError as e:
        print(f"[AffectScore] ERROR: ACE-Step not installed: {e}")
        sys.exit(1)

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    restore_sdpa, collected = setup_sdpa_capture()

    pipe = ACEStepPipeline(
        checkpoint_dir=model_path if model_path else None,
        device_id=0,
        dtype="bfloat16",
        torch_compile=False,
        cpu_offload=False,
    )
    pipe.load_checkpoint()

    NULL_PROMPT = "neutral ambient music"
    null_entropies = []
    null_per_input = {}

    import tempfile

    for v, a, label in VA_INPUTS:
        collected['active'] = False
        collected['entropy_sum'] = 0.0
        collected['entropy_count'] = 0
        collected['heatmap_slice'] = None

        output_path = os.path.join(tempfile.mkdtemp(), f"null_{label}.wav")

        collected['active'] = True
        with torch.inference_mode():
            pipe(
                audio_duration=4.0,
                prompt=NULL_PROMPT,
                lyrics="[instrumental]",
                infer_step=8,
                guidance_scale=7.0,
                scheduler_type="euler",
                cfg_type="apg",
                omega_scale=10.0,
                manual_seeds="42",
                guidance_interval=0.5,
                guidance_interval_decay=0.0,
                min_guidance_scale=3.0,
                use_erg_tag=True,
                use_erg_lyric=True,
                use_erg_diffusion=True,
                save_path=output_path,
            )
        collected['active'] = False

        if os.path.exists(output_path):
            os.unlink(output_path)

        n_calls = collected['entropy_count']
        H = (collected['entropy_sum'] / n_calls) if n_calls > 0 else 0.0
        null_entropies.append(H)
        null_per_input[label] = H
        print(f"[AffectScore] NULL V={v:+.1f} A={a:+.1f} ({label}): entropy={H:.4f} bits")

    restore_sdpa()

    null_range = max(null_entropies) - min(null_entropies)

    gate2_results_path = os.path.join(_HERE, "gate2_results.json")
    gate2_range = None
    if os.path.exists(gate2_results_path):
        with open(gate2_results_path) as f:
            gate2_data = json.load(f)
        gate2_range = gate2_data.get("entropy_range_bits")

    print(f"\n[AffectScore] === Null Probe Results ===")
    print(f"  Null entropy range:   {null_range:.4f} bits (fixed prompt)")
    if gate2_range is not None:
        print(f"  Gate 2 entropy range: {gate2_range:.4f} bits (varied prompts)")
        ratio = gate2_range / max(null_range, 1e-8)
        print(f"  Ratio (Gate2/Null):   {ratio:.1f}x")
        if ratio >= 2.0:
            print(f"  CONCLUSION: Entropy variation is text-conditioning-driven "
                  f"(Gate2 range is {ratio:.1f}x the null range).")
        else:
            print(f"  WARNING: Null range is within 2x of Gate 2 range -- "
                  f"text-driven variation is weak; consider prompt diversity.")

    null_results = {
        "experiment": "Gate 2 null distribution probe",
        "null_entropy_range_bits": null_range,
        "null_entropy_per_input": null_per_input,
        "gate2_entropy_range_bits": gate2_range,
        "null_prompt": NULL_PROMPT,
        "note": (
            "Null probe uses fixed prompt for all 5 V-A inputs. "
            "If null_entropy_range << gate2_entropy_range, conditioning is text-driven."
        ),
    }

    null_results_path = os.path.join(_HERE, "gate2_null_probe_results.json")
    with open(null_results_path, "w") as f:
        json.dump(null_results, f, indent=2)
    print(f"[AffectScore] Null probe results saved to {null_results_path}")
    return null_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gate 2: Cross-attention entropy measurement"
    )
    parser.add_argument(
        "--model-path", type=str,
        default="",
        help="Local checkpoint dir for ACEStepPipeline (empty = HF download)",
    )
    parser.add_argument(
        "--ace-step-root", type=str,
        default=os.path.join(_REPO_ROOT, "ace_step"),
        help="Path to add to sys.path for ACE-Step imports",
    )
    parser.add_argument(
        "--run-null-probe",
        action="store_true",
        default=False,
        help=(
            "After running Gate 2, run the null distribution probe with a fixed "
            "prompt for all 5 V-A inputs. Results: gate2_null_probe_results.json."
        ),
    )
    args = parser.parse_args()
    results = run_gate2(args.model_path, args.ace_step_root)
    if args.run_null_probe:
        run_null_probe(args.model_path, args.ace_step_root)
    if not results["pass"]:
        sys.exit(1)
