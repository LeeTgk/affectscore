"""
MER accuracy evaluation using Music2Emo (MERT-v1-95M backbone) on generated audio.
Computes Pearson r and RMSE between designer-intended V-A and Music2Emo-predicted V-A.
Distinct from CLAP-score.

Usage:
    python eval/emotion_classify.py \\
        --audio-dir /path/to/generated_wavs/ \\
        --held-out data/held_out_set.json \\
        --rank 32
"""

import os
import sys
import json
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_M2E_REPO = os.path.join(_REPO_ROOT, "music2emo_repo")
_M2E_WEIGHTS = os.path.join(_M2E_REPO, "saved_models", "J_all.ckpt")


def _load_model():
    """Import Music2emo and instantiate it.

    Must be called after os.chdir(_M2E_REPO) because music2emo.py and its
    predict() method hard-code relative paths to inference/data/*.

    torchaudio>=2.9 delegates torchaudio.load() to TorchCodec which requires
    FFmpeg DLLs (unavailable here).  We monkey-patch it with a librosa-based
    implementation that reads WAV/MP3 natively via soundfile.
    """
    import numpy as np
    import torch
    import torchaudio
    import librosa

    def _librosa_load(path, frame_offset=0, num_frames=-1, normalize=True,
                      channels_first=True, format=None, buffer_size=4096, backend=None):
        y, sr = librosa.load(str(path), sr=None, mono=False)
        if y.ndim == 1:
            y = y[np.newaxis, :]  # (1, n_samples)
        return torch.from_numpy(y.astype(np.float32)), int(sr)

    torchaudio.load = _librosa_load

    # PyTorch 2.6 changed torch.load default to weights_only=True, breaking
    # music2emo's chord model checkpoints (contain numpy globals).
    # Force weights_only=False for the duration of music2emo calls.
    # Safe: these are trusted AMAAI-Lab checkpoints on a closed Colab session.
    _orig_torch_load = torch.load
    def _torch_load_unsafe(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _torch_load_unsafe

    if _M2E_REPO not in sys.path:
        sys.path.insert(0, _M2E_REPO)
    from music2emo import Music2emo
    return Music2emo(model_weights=_M2E_WEIGHTS)


def _normalize_va(raw_v: float, raw_a: float):
    """Convert Music2Emo 1-9 DEAM scale to [-1, 1] range.

    Music2Emo outputs valence/arousal on a 1-9 scale (5.0 = center).
    manifest.json uses [-1, 1] range (0.0 = center).
    """
    valence = max(-1.0, min(1.0, (raw_v - 5.0) / 4.0))
    arousal = max(-1.0, min(1.0, (raw_a - 5.0) / 4.0))
    return valence, arousal


def run_emotion_classify(held_out_path: str, audio_dir: str, rank: int = None):
    """Run Music2Emo on generated audio and compute Pearson r + RMSE against intended V-A.

    Args:
        held_out_path: Path to data/held_out_set.json
        audio_dir: Directory of generated WAV files (named by clip_id)
        rank: Optional LoRA rank being evaluated (for result file naming)

    Returns:
        Dict with MER accuracy results.
    """
    print(f"[AffectScore] Loading held-out set from {held_out_path}")
    with open(held_out_path) as f:
        held_out = json.load(f)

    print(f"[AffectScore] Held-out clips: {len(held_out)}")
    print(f"[AffectScore] Audio directory: {audio_dir}")
    if rank is not None:
        print(f"[AffectScore] Evaluating LoRA rank r={rank}")

    print("[AffectScore] Loading Music2Emo model (MERT + chord, ~30s first load)...")
    orig_cwd = os.getcwd()
    # MUST chdir to _M2E_REPO before loading -- music2emo.py uses relative paths
    os.chdir(_M2E_REPO)
    try:
        model = _load_model()
        print("[AffectScore] Music2Emo loaded.")

        intended_v, intended_a = [], []
        predicted_v, predicted_a = [], []
        per_clip_results = []
        skipped = 0

        for i, clip in enumerate(held_out):
            clip_id = clip["clip_id"]
            wav_path = os.path.join(audio_dir, f"{clip_id}.wav")
            if not os.path.exists(wav_path):
                wav_path_alt = os.path.join(audio_dir, f"{clip_id}_generated.wav")
                if os.path.exists(wav_path_alt):
                    wav_path = wav_path_alt
                else:
                    skipped += 1
                    continue

            try:
                output = model.predict(wav_path)
                raw_v = output.get("valence", 5.0)
                raw_a = output.get("arousal", 5.0)
                pred_v, pred_a = _normalize_va(raw_v, raw_a)
            except Exception as e:
                print(f"[AffectScore] Music2Emo failed on {clip_id}: {e}")
                skipped += 1
                continue

            int_v = clip.get("V_A_valence", 0.0)
            int_a = clip.get("V_A_arousal", 0.0)

            intended_v.append(int_v)
            intended_a.append(int_a)
            predicted_v.append(pred_v)
            predicted_a.append(pred_a)

            per_clip_results.append({
                "clip_id": clip_id,
                "intended_valence": int_v,
                "intended_arousal": int_a,
                "predicted_valence": pred_v,
                "predicted_arousal": pred_a,
            })

            if (i + 1) % 20 == 0:
                print(f"[AffectScore] Processed {i + 1}/{len(held_out)} clips "
                      f"({skipped} skipped)...")

    finally:
        os.chdir(orig_cwd)

    if not intended_v:
        print(f"[AffectScore] ERROR: No clips evaluated. Skipped: {skipped}")
        return {"error": "No clips evaluated", "skipped": skipped}

    import numpy as np
    from scipy.stats import pearsonr

    iv = np.array(intended_v)
    ia = np.array(intended_a)
    pv = np.array(predicted_v)
    pa = np.array(predicted_a)

    R_v, _ = pearsonr(iv, pv)
    R_a, _ = pearsonr(ia, pa)
    RMSE_v = float(np.sqrt(np.mean((iv - pv) ** 2)))
    RMSE_a = float(np.sqrt(np.mean((ia - pa) ** 2)))

    results = {
        "rank": rank,
        "n_clips_evaluated": len(intended_v),
        "n_clips_skipped": skipped,
        "mer_accuracy": {
            "pearson_r_valence": float(R_v),
            "pearson_r_arousal": float(R_a),
            "rmse_valence": RMSE_v,
            "rmse_arousal": RMSE_a,
        },
        "metric_description": (
            "MER accuracy: Pearson r and RMSE between designer-intended V-A "
            "(held_out_set.json MERT-auto labels) and Music2Emo-predicted V-A "
            "from generated audio."
        ),
        "per_clip_results": per_clip_results,
    }

    print(f"\n[AffectScore] === MER Accuracy Summary (rank={rank}) ===")
    print(f"  Clips evaluated: {len(intended_v)} / {len(held_out)} ({skipped} skipped)")
    print(f"  Pearson r:  valence={R_v:.3f}, arousal={R_a:.3f}")
    print(f"  RMSE:       valence={RMSE_v:.3f}, arousal={RMSE_a:.3f}")

    rank_tag = f"_r{rank}" if rank is not None else ""
    results_filename = f"emotion_classify_results{rank_tag}.json"
    results_path = os.path.join(_HERE, results_filename)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[AffectScore] Results saved to {results_path}")

    return results


def _assign_quadrant(valence: float, arousal: float) -> str:
    """Assign a Russell circumplex quadrant label from V-A values.

    Uses the same 0.3 threshold as va_to_mood_words for consistency.
    Returns 'Q1' (high V, high A), 'Q2' (low V, high A),
    'Q3' (low V, low A), 'Q4' (high V, low A), or 'QC' (center).
    """
    if valence >= 0.3 and arousal >= 0.3:
        return "Q1"
    elif valence < -0.3 and arousal >= 0.3:
        return "Q2"
    elif valence < -0.3 and arousal < -0.3:
        return "Q3"
    elif valence >= 0.3 and arousal < -0.3:
        return "Q4"
    else:
        return "QC"


def run_emotion_classify_stratified(
    held_out_path: str,
    audio_dir: str,
    rank: int = None,
    seed: int = 42,
) -> dict:
    """Quadrant-stratified MER on a Q2-matched balanced subset.

    The full held-out set is Q2-dominant in the wrong direction: Q2 (tense,
    ~22% held-out) vs. only ~2.2% Q2 in training. This confounds the full-set
    MER metric because the evaluation set is disproportionately rich in the
    quadrant most underrepresented during training.

    Balances by sampling min(quadrant count) clips from each Russell quadrant,
    removing the training--evaluation distribution mismatch confound and giving
    a cleaner estimate of directional V-A conditioning on equal-class data.

    Args:
        held_out_path: Path to data/held_out_set.json.
        audio_dir: Directory of generated WAV files (named {clip_id}.wav).
        rank: Optional LoRA rank tag for result file naming.
        seed: Random seed for quadrant-balanced sampling (default 42).

    Returns:
        Dict with full results including per-quadrant breakdown and balanced metrics.
    """
    import numpy as np
    from scipy.stats import pearsonr

    print(f"[AffectScore] === Stratified MER (quadrant-balanced subset) ===")
    print(f"[AffectScore] Audio dir: {audio_dir}")
    print(f"[AffectScore] Seed: {seed}")

    with open(held_out_path) as f:
        held_out = json.load(f)

    quadrant_clips = {"Q1": [], "Q2": [], "Q3": [], "Q4": [], "QC": []}
    for clip in held_out:
        v = clip.get("V_A_valence", 0.0)
        a = clip.get("V_A_arousal", 0.0)
        wav_path = os.path.join(audio_dir, f"{clip['clip_id']}.wav")
        if not os.path.exists(wav_path):
            alt = os.path.join(audio_dir, f"{clip['clip_id']}_generated.wav")
            if os.path.exists(alt):
                wav_path = alt
            else:
                continue
        q = _assign_quadrant(v, a)
        quadrant_clips[q].append({**clip, "_wav_path": wav_path})

    print(f"[AffectScore] Quadrant counts before balancing:")
    for q, clips in quadrant_clips.items():
        print(f"[AffectScore]   {q}: {len(clips)} clips")

    non_empty = {q: c for q, c in quadrant_clips.items() if c}
    if len(non_empty) < 2:
        return {
            "error": "Too few quadrants with audio to balance",
            "counts": {q: len(c) for q, c in quadrant_clips.items()},
        }

    min_count = min(len(c) for c in non_empty.values())
    print(f"[AffectScore] Limiting quadrant: {min_count} clips. "
          f"Sampling {min_count} from each non-empty quadrant.")

    rng = np.random.default_rng(seed)
    balanced_clips = []
    for q, clips in non_empty.items():
        sampled = rng.choice(len(clips), size=min(min_count, len(clips)), replace=False).tolist()
        balanced_clips.extend([clips[i] for i in sampled])

    print(f"[AffectScore] Balanced subset: {len(balanced_clips)} clips "
          f"({len(non_empty)} quadrants x {min_count})")

    orig_cwd = os.getcwd()
    os.chdir(_M2E_REPO)
    try:
        model = _load_model()

        intended_v, intended_a = [], []
        predicted_v, predicted_a = [], []
        per_clip_results = []
        skipped = 0

        for clip in balanced_clips:
            wav_path = clip["_wav_path"]
            try:
                output = model.predict(wav_path)
                pred_v, pred_a = _normalize_va(
                    output.get("valence", 5.0), output.get("arousal", 5.0)
                )
            except Exception as e:
                print(f"[AffectScore] Music2Emo failed on {clip['clip_id']}: {e}")
                skipped += 1
                continue

            int_v = clip.get("V_A_valence", 0.0)
            int_a = clip.get("V_A_arousal", 0.0)
            intended_v.append(int_v)
            intended_a.append(int_a)
            predicted_v.append(pred_v)
            predicted_a.append(pred_a)
            per_clip_results.append({
                "clip_id": clip["clip_id"],
                "quadrant": _assign_quadrant(int_v, int_a),
                "intended_valence": int_v,
                "intended_arousal": int_a,
                "predicted_valence": pred_v,
                "predicted_arousal": pred_a,
            })

    finally:
        os.chdir(orig_cwd)

    if not intended_v:
        return {"error": "No clips evaluated in balanced subset", "skipped": skipped}

    iv = np.array(intended_v)
    ia = np.array(intended_a)
    pv = np.array(predicted_v)
    pa = np.array(predicted_a)

    R_v, p_v = pearsonr(iv, pv)
    R_a, p_a = pearsonr(ia, pa)
    RMSE_v = float(np.sqrt(np.mean((iv - pv) ** 2)))
    RMSE_a = float(np.sqrt(np.mean((ia - pa) ** 2)))

    print(f"\n[AffectScore] === Stratified MER Summary (rank={rank}) ===")
    print(f"  Balanced subset: {len(intended_v)} clips ({skipped} skipped)")
    print(f"  Pearson r: valence={R_v:.3f} (p={p_v:.4f}), arousal={R_a:.3f} (p={p_a:.4f})")
    print(f"  RMSE:      valence={RMSE_v:.3f}, arousal={RMSE_a:.3f}")

    results = {
        "experiment": "quadrant-stratified MER (balanced subset)",
        "rank": rank,
        "seed": seed,
        "quadrant_counts_before_balance": {q: len(c) for q, c in quadrant_clips.items()},
        "min_count_per_quadrant": min_count,
        "n_clips_evaluated": len(intended_v),
        "n_clips_skipped": skipped,
        "mer_accuracy_balanced": {
            "pearson_r_valence": float(R_v),
            "pearson_p_valence": float(p_v),
            "pearson_r_arousal": float(R_a),
            "pearson_p_arousal": float(p_a),
            "rmse_valence": RMSE_v,
            "rmse_arousal": RMSE_a,
        },
        "per_clip_results": per_clip_results,
        "note": (
            "Balanced subset removes the training--evaluation distribution mismatch "
            "confound (Q2 over-represented in held-out at ~22% vs 2.2% in training). "
            "Compare against run_emotion_classify() full-set results for the same adapter."
        ),
    }

    rank_tag = f"_r{rank}" if rank is not None else ""
    results_path = os.path.join(_HERE, f"emotion_classify_stratified_results{rank_tag}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[AffectScore] Stratified results saved to {results_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MER accuracy evaluation for LoRA rank selection via Music2Emo"
    )
    parser.add_argument(
        "--held-out", type=str,
        default=os.path.join(_REPO_ROOT, "data", "held_out_set.json"),
        help="Path to held_out_set.json (default: data/held_out_set.json)",
    )
    parser.add_argument(
        "--audio-dir", type=str, required=True,
        help="Directory of generated WAV files (named {clip_id}.wav)",
    )
    parser.add_argument(
        "--rank", type=int, default=None,
        help="LoRA rank being evaluated (e.g., 16, 32, 64) -- for result file naming",
    )
    parser.add_argument(
        "--stratified",
        action="store_true",
        default=False,
        help=(
            "Run quadrant-stratified (balanced) MER in addition to the full-set "
            "evaluation. Samples min(quadrant count) clips from each Russell quadrant "
            "to remove the Q2 over-representation confound. Results saved separately."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for quadrant-balanced sampling (--stratified only, default 42).",
    )
    args = parser.parse_args()
    run_emotion_classify(args.held_out, args.audio_dir, args.rank)
    if args.stratified:
        run_emotion_classify_stratified(args.held_out, args.audio_dir, args.rank, args.seed)
