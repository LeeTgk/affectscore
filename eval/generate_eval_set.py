"""
Shared audio generation pipeline for AffectScore evaluation.
Generates 248 WAV clips per adapter variant (200 held-out + 48 archetype).

Usage (Colab A100):
    python eval/generate_eval_set.py \\
        --adapter-name full \\
        --adapter-path /content/drive/MyDrive/affectscore/checkpoints/ace-step-r32-20260701/final_adapter \\
        --drive-output /content/drive/MyDrive/affectscore/eval_outputs

For no-lora (base model inference), omit --adapter-path:
    python eval/generate_eval_set.py \\
        --adapter-name no-lora \\
        --drive-output /content/drive/MyDrive/affectscore/eval_outputs
"""

import os
import sys
import json
import argparse
import struct
import wave
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_HERE, "results")

N_HELD_OUT = 200
N_ARCHETYPE = 12
ADAPTER_VARIANTS = ["full", "no-lora", "no-affect", "no-style", "r16", "r64"]

ARCHETYPES = {
    "contemplative": {
        "arc_position": 0.3,
        "choice_latency_norm": 0.8,
        "dwell_deviation_norm": 0.7,
        "interaction_rate_norm": 0.2,
    },
    "impulsive": {
        "arc_position": 0.3,
        "choice_latency_norm": 0.1,
        "dwell_deviation_norm": 0.2,
        "interaction_rate_norm": 0.9,
    },
    "tense": {
        "arc_position": 0.7,
        "choice_latency_norm": 0.6,
        "dwell_deviation_norm": 0.3,
        "interaction_rate_norm": 0.5,
    },
    "neutral": {
        "arc_position": 0.5,
        "choice_latency_norm": 0.5,
        "dwell_deviation_norm": 0.5,
        "interaction_rate_norm": 0.5,
    },
}

DESIGNER_VALENCE = 0.0   # fixed neutral anchor
DESIGNER_AROUSAL = 0.0   # fixed neutral anchor
ARCHETYPE_STYLE_PROMPT = "ambient orchestral game soundtrack neutral calm"

# Controlled archetype profiles -- all arc_position=0.5.
# Engagement signals are identical to ARCHETYPES; only arc_position is held constant.
# Used with --controlled flag.
ARCHETYPES_CONTROLLED = {
    "contemplative": {
        "arc_position": 0.5,
        "choice_latency_norm": 0.8,
        "dwell_deviation_norm": 0.7,
        "interaction_rate_norm": 0.2,
    },
    "impulsive": {
        "arc_position": 0.5,
        "choice_latency_norm": 0.1,
        "dwell_deviation_norm": 0.2,
        "interaction_rate_norm": 0.9,
    },
    "tense": {
        "arc_position": 0.5,
        "choice_latency_norm": 0.6,
        "dwell_deviation_norm": 0.3,
        "interaction_rate_norm": 0.5,
    },
    "neutral": {
        "arc_position": 0.5,
        "choice_latency_norm": 0.5,
        "dwell_deviation_norm": 0.5,
        "interaction_rate_norm": 0.5,
    },
}


def va_to_mood_words(valence: float, arousal: float) -> str:
    """Map V-A values to CLAP-compatible text string.

    Verbatim copy from server/affectscore_server.py. Do NOT import from server/
    (process-separation constraint -- no server imports in eval scripts).

    Uses Russell circumplex quadrant boundaries at threshold 0.3.
    V in [-1, 1], A in [-1, 1] (normalised MERT-auto scale).

    Vocabulary and thresholds must match server/affectscore_server.py verbatim --
    the text seen by ACE-Step at inference must match what was used during training.
    """
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


def engagement_to_modifier_words(
    choice_latency_norm: float,
    dwell_deviation_norm: float,
    interaction_rate_norm: float,
    arc_position: float = 0.5,
) -> str:
    """Map normalised player engagement signals to CLAP-compatible style modifiers.

    Thresholds must match server/affectscore_server.py verbatim.
    Returns empty string for mid-range inputs (neutral archetype, all signals 0.5).
    Thresholds at 0.65 / 0.35 follow the va_to_mood_words() boundary design.

    Args:
        choice_latency_norm: 0 = fast decisions, 1 = slow/deliberate.
        dwell_deviation_norm: 0 = under-dwells, 1 = over-dwells/immersed.
        interaction_rate_norm: 0 = few interactions, 1 = many interactions.
        arc_position: 0 = story start, 1 = story end.
    """
    modifiers = []

    if interaction_rate_norm >= 0.65:
        modifiers.append("energetic driven")
    elif interaction_rate_norm <= 0.35:
        modifiers.append("unhurried contemplative")

    if choice_latency_norm >= 0.65:
        modifiers.append("deliberate")
    elif choice_latency_norm <= 0.35:
        modifiers.append("impulsive")

    if dwell_deviation_norm >= 0.65:
        modifiers.append("immersive")
    elif dwell_deviation_norm <= 0.35:
        modifiers.append("transient")

    if arc_position >= 0.65:
        modifiers.append("building")

    return " ".join(modifiers)


def _ensure_diffusers_lora_filename(adapter_dir: str) -> None:
    """ACEStepPipeline.load_lora hardcodes 'pytorch_lora_weights.safetensors'.
    PEFT saves as 'adapter_model.safetensors'. If the diffusers name is missing
    but the PEFT name exists, rename it in-place so load_lora can find it.
    """
    target = os.path.join(adapter_dir, "pytorch_lora_weights.safetensors")
    peft_src = os.path.join(adapter_dir, "adapter_model.safetensors")
    peft_bin = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.exists(target):
        return
    if os.path.exists(peft_src):
        print(f"[AffectScore] Renaming adapter_model.safetensors -> pytorch_lora_weights.safetensors in {adapter_dir}")
        os.rename(peft_src, target)
    elif os.path.exists(peft_bin):
        print(f"[AffectScore] WARNING: adapter_model.bin found but only .safetensors supported. Convert first.")
    else:
        raise FileNotFoundError(
            f"[AffectScore] No LoRA weights found in {adapter_dir}. "
            f"Expected pytorch_lora_weights.safetensors or adapter_model.safetensors."
        )


def generate_single_clip(
    clip_id: str,
    output_dir: str,
    pipeline,
    valence: float = 0.0,
    arousal: float = 0.0,
    style_prompt: str = "neutral ambient music",
    num_inference_steps: int = 40,
    duration_s: float = 4.0,
    lora_name_or_path: str = None,
) -> bool:
    """Generate a single 4s WAV clip and write to output_dir/{clip_id}.wav.

    Returns True if a new file was generated, False if an existing valid
    file was found and skipped (caching guard).

    Caching guard: if the output file already exists AND is >1 KB, the clip is
    considered complete and generation is skipped. This allows interrupted runs
    to resume without re-generating clips.

    Args:
        clip_id: Unique identifier for the clip (used as filename stem).
        output_dir: Directory to write the WAV file to.
        pipeline: An ACEStepPipeline instance (or compatible mock).
        valence: Designer-intent valence [-1, 1].
        arousal: Designer-intent arousal [-1, 1].
        style_prompt: Text conditioning prompt for the pipeline.
        num_inference_steps: Diffusion steps (40 for evaluation standard).
        duration_s: Output duration in seconds (4.0 for evaluation standard).
        lora_name_or_path: Adapter path or "none" for base model inference.

    Returns:
        True if clip was generated, False if skipped.
    """
    wav_path = Path(output_dir) / f"{clip_id}.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    if wav_path.exists() and wav_path.stat().st_size > 1000:
        print(
            f"[AffectScore]   [skip] {clip_id} already exists "
            f"({wav_path.stat().st_size} bytes)"
        )
        return False

    # ACEStepPipeline.__call__ saves WAV via torchaudio and returns
    # [saved_path, ..., input_params_json]. Pass save_path so it writes to our
    # controlled location. lyrics="" = instrumental (len("") == 0 skips lyric
    # tokenisation; passing None would crash at len(None)).
    kwargs = {
        "audio_duration": duration_s,
        "infer_step": num_inference_steps,
        "prompt": style_prompt,
        "lyrics": "",
        "save_path": str(wav_path),
        "lora_name_or_path": lora_name_or_path if lora_name_or_path is not None else "none",
        "lora_weight": 1.0,
    }

    pipeline(**kwargs)
    return True


def generate_eval_set(
    held_out_path: str,
    adapter_name: str,
    adapter_path: str,
    drive_output_base: str,
    results_dir: str = None,
    num_inference_steps: int = 40,
    duration_s: float = 4.0,
    output_tag: str = "",
    controlled: bool = False,
) -> dict:
    """Generate the full 248-clip evaluation set for one adapter variant.

    Produces:
      - 200 held-out clips in {drive_output_base}/{adapter_name}{_tag}/{clip_id}.wav
      - 48 archetype clips (4 archetypes x 12 clips) in same output dir
      - eval/results/eval_manifest_{adapter_name}{_tag}.json

    Args:
        held_out_path: Path to data/held_out_set.json
        adapter_name: One of ADAPTER_VARIANTS. Validated against allowlist.
        adapter_path: Path to adapter directory on Drive. Pass "none" (string)
            for no-lora (base model inference). Required for all other variants.
        drive_output_base: Base path on Drive for WAV output.
        results_dir: Directory for manifest JSON. Defaults to eval/results/.
        num_inference_steps: Inference steps per clip (default 40).
        duration_s: Clip duration in seconds (default 4.0).
        output_tag: Optional suffix appended to output dir and manifest name.
            Use 'step8' when running with steps=8 to avoid overwriting 40-step
            outputs, e.g. adapter 'full' + tag 'step8' -> 'full_step8/' dir.
        controlled: If True, use ARCHETYPES_CONTROLLED (all arc_position=0.5)
            instead of ARCHETYPES for archetype clip generation.
            Output dir is tagged '_ctrl' unless output_tag is set.

    Returns:
        Manifest dict written to eval_manifest_{adapter_name}{_tag}.json.
    """
    if adapter_name not in ADAPTER_VARIANTS:
        raise ValueError(
            f"[AffectScore] Unknown adapter_name '{adapter_name}'. "
            f"Must be one of {ADAPTER_VARIANTS}"
        )

    if results_dir is None:
        results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    if output_tag:
        dir_suffix = f"_{output_tag}"
    elif controlled:
        dir_suffix = "_ctrl"
    else:
        dir_suffix = ""
    dir_name = f"{adapter_name}{dir_suffix}"

    output_dir = os.path.join(os.path.abspath(drive_output_base), dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # "no-lora" = base model, pass lora_name_or_path="none" to unload any loaded adapter.
    if adapter_name == "no-lora" or str(adapter_path).lower() == "none":
        lora_name_or_path = "none"
    else:
        lora_name_or_path = os.path.abspath(adapter_path)
        _ensure_diffusers_lora_filename(lora_name_or_path)

    print(f"[AffectScore] Initialising ACEStepPipeline (adapter_name={adapter_name})...")
    import torch
    from acestep.pipeline_ace_step import ACEStepPipeline

    # ACEStepPipeline uses a plain __init__, not from_pretrained.
    # checkpoint_dir=None -> auto-downloads to ~/.cache/ace-step/ (symlinked to Drive).
    pipe = ACEStepPipeline(
        checkpoint_dir=None,
        device_id=0,
        dtype="bfloat16",
    )
    print("[AffectScore] Pipeline ready.")

    held_out_path = os.path.abspath(held_out_path)
    print(f"[AffectScore] Loading held-out set from {held_out_path}")
    with open(held_out_path) as f:
        held_out = json.load(f)

    manifest_entries = []
    total_clips = N_HELD_OUT + len(ARCHETYPES) * N_ARCHETYPE  # 248
    clip_count = 0
    skipped = 0

    print(
        f"[AffectScore] Generating {N_HELD_OUT} held-out clips "
        f"(adapter={adapter_name}, steps={num_inference_steps}) ..."
    )
    for clip in held_out[:N_HELD_OUT]:
        clip_id = clip["clip_id"]
        valence = clip.get("V_A_valence", 0.0)
        arousal = clip.get("V_A_arousal", 0.0)

        # Style prompt derived from V-A via va_to_mood_words.
        # held_out_set.json has no "style_prompt" field -- must derive at eval time.
        mood_words = va_to_mood_words(valence, arousal)
        style_prompt = mood_words + " orchestral game soundtrack"

        generated = generate_single_clip(
            clip_id=clip_id,
            output_dir=output_dir,
            pipeline=pipe,
            valence=valence,
            arousal=arousal,
            style_prompt=style_prompt,
            num_inference_steps=num_inference_steps,
            duration_s=duration_s,
            lora_name_or_path=lora_name_or_path,
        )
        if not generated:
            skipped += 1

        clip_count += 1
        manifest_entries.append({
            "clip_id": clip_id,
            "clip_type": "held_out",
            "style_prompt": style_prompt,
            "V_A_valence": valence,
            "V_A_arousal": arousal,
            "wav_path": str(Path(output_dir) / f"{clip_id}.wav"),
        })

        if clip_count % 20 == 0:
            print(
                f"[AffectScore] Progress: {clip_count}/{total_clips} "
                f"({skipped} skipped)"
            )

    archetype_dict = ARCHETYPES_CONTROLLED if controlled else ARCHETYPES
    if controlled:
        print(
            f"[AffectScore] Controlled mode: using ARCHETYPES_CONTROLLED "
            f"(all arc_position=0.5)."
        )
    print(
        f"[AffectScore] Generating {len(archetype_dict) * N_ARCHETYPE} archetype clips ..."
    )
    archetype_prefix_map = {
        "contemplative": "contemp",
        "impulsive": "impulsive",
        "tense": "tense",
        "neutral": "neutral",
    }
    for archetype_name, signals in archetype_dict.items():
        prefix = archetype_prefix_map.get(archetype_name, archetype_name)
        # Archetype clips go into a per-archetype subdirectory so that
        # simulated_archetypes.py can find them via {base}/{archetype_name}/.
        archetype_dir = Path(output_dir) / archetype_name
        archetype_dir.mkdir(parents=True, exist_ok=True)

        # Each archetype gets a distinct style_prompt so FAD can measure whether
        # text-space Layer 2 variation produces separable distributions.
        engagement_modifier = engagement_to_modifier_words(
            choice_latency_norm=signals["choice_latency_norm"],
            dwell_deviation_norm=signals["dwell_deviation_norm"],
            interaction_rate_norm=signals["interaction_rate_norm"],
            arc_position=signals["arc_position"],
        )
        style_prompt = (
            f"{ARCHETYPE_STYLE_PROMPT} {engagement_modifier}".strip()
        )
        print(
            f"[AffectScore] Archetype '{archetype_name}': "
            f"engagement_modifier='{engagement_modifier}' -> "
            f"style_prompt='{style_prompt}'"
        )

        for i in range(N_ARCHETYPE):
            clip_id = f"{prefix}_{i:03d}"

            generated = generate_single_clip(
                clip_id=clip_id,
                output_dir=str(archetype_dir),
                pipeline=pipe,
                valence=DESIGNER_VALENCE,
                arousal=DESIGNER_AROUSAL,
                style_prompt=style_prompt,
                num_inference_steps=num_inference_steps,
                duration_s=duration_s,
                lora_name_or_path=lora_name_or_path,
            )
            if not generated:
                skipped += 1

            clip_count += 1
            manifest_entries.append({
                "clip_id": clip_id,
                "clip_type": "archetype",
                "archetype": archetype_name,
                "style_prompt": style_prompt,
                "engagement_modifier": engagement_modifier,
                "V_A_valence": DESIGNER_VALENCE,
                "V_A_arousal": DESIGNER_AROUSAL,
                "engagement_signals": signals,
                "wav_path": str(archetype_dir / f"{clip_id}.wav"),
            })

            if clip_count % 20 == 0:
                print(
                    f"[AffectScore] Progress: {clip_count}/{total_clips} "
                    f"({skipped} skipped)"
                )

    manifest = {
        "adapter_name": adapter_name,
        "num_inference_steps": num_inference_steps,
        "duration_s": duration_s,
        "lora_name_or_path": lora_name_or_path,
        "output_dir": output_dir,
        "n_generated": clip_count - skipped,
        "n_skipped": skipped,
        "n_total": clip_count,
        "clips": manifest_entries,
    }

    manifest_path = os.path.join(
        results_dir, f"eval_manifest_{dir_name}.json"
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[AffectScore] Manifest written to {manifest_path}")
    print(
        f"[AffectScore] Done. {clip_count} clips total "
        f"({clip_count - skipped} generated, {skipped} skipped)."
    )

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 248 evaluation WAVs per adapter variant for AffectScore."
    )
    parser.add_argument(
        "--held-out",
        type=str,
        default=os.path.join(_REPO_ROOT, "data", "held_out_set.json"),
        help="Path to data/held_out_set.json (default: %(default)s)",
    )
    parser.add_argument(
        "--adapter-name",
        type=str,
        required=True,
        choices=ADAPTER_VARIANTS,
        help=f"Adapter variant to generate for. One of: {ADAPTER_VARIANTS}",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="none",
        help=(
            "Path to adapter directory on Drive. "
            "Required for all variants except no-lora. "
            "Pass 'none' or omit for no-lora (base model inference)."
        ),
    )
    parser.add_argument(
        "--drive-output",
        type=str,
        required=True,
        help=(
            "Drive output base path, e.g. "
            "/content/drive/MyDrive/affectscore/eval_outputs"
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help=(
            "Inference steps per clip (default: %(default)s). "
            "Use --steps 8 with --output-tag step8 for 8-step evaluation."
        ),
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help=(
            "Optional suffix appended to output directory and manifest filename. "
            "Example: --steps 8 --output-tag step8 writes to {adapter}_step8/ and "
            "eval_manifest_{adapter}_step8.json."
        ),
    )
    parser.add_argument(
        "--controlled",
        action="store_true",
        default=False,
        help=(
            "Use ARCHETYPES_CONTROLLED (all arc_position=0.5) for archetype clip generation. "
            "Output dir is tagged '_ctrl' unless --output-tag is also set."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help="Directory for manifest JSON (default: %(default)s)",
    )
    args = parser.parse_args()

    generate_eval_set(
        held_out_path=args.held_out,
        adapter_name=args.adapter_name,
        adapter_path=args.adapter_path,
        drive_output_base=args.drive_output,
        results_dir=args.results_dir,
        num_inference_steps=args.steps,
        output_tag=args.output_tag,
        controlled=args.controlled,
    )
