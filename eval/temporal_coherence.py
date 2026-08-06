"""
Temporal coherence evaluation for AffectScore.
Measures Log-Spectral Distance (LSD) at chunk boundaries via 15 sequential
/generate calls, comparing hard-cut and crossfaded conditions against a natural
within-track baseline using Wilcoxon rank-sum tests.

Hard-cut is the primary measure: it isolates model-level coherence without
post-processing smoothing. Crossfaded is the deployable delivery format.

Usage (Colab A100):
    # 1. Start server with target adapter:
    #    python server/affectscore_server.py --lora /path/to/adapter &
    # 2. Wait for /health to return 200
    # 3. Run:
    python eval/temporal_coherence.py \\
        --adapter-name full \\
        --held-out data/held_out_set.json
"""

import os
import sys
import json
import argparse
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

SERVER_URL = "http://127.0.0.1:8321"
N_CHUNKS = 15                  # Sequential /generate calls per condition
LSD_N_FFT = 2048               # STFT window size for LSD computation
LSD_HOP_LENGTH = 512           # STFT hop length for LSD computation
LSD_CONTEXT_FRAMES = 5         # Frames on each side of boundary for LSD
CROSSFADE_DURATION_S = 0.5
SR_LOAD = 48000                # ACE-Step output sample rate
SR_SPEC = 22050                # Downsampled rate for spectrogram display
N_BASELINE_CLIPS = 200         # Held-out clips for natural baseline
N_BASELINE_SPLITS = 3          # Split points per clip (25%, 50%, 75%)

RESULTS_DIR = os.path.join(_HERE, "results")
DOCS_FIGURES = os.path.join(_REPO_ROOT, "docs", "figures")


def compute_lsd_at_boundary(
    chunk_a: np.ndarray,
    chunk_b: np.ndarray,
    sr: int = 48000,
    n_fft: int = LSD_N_FFT,
    hop_length: int = LSD_HOP_LENGTH,
    context_frames: int = LSD_CONTEXT_FRAMES,
) -> float:
    """Log-Spectral Distance at a chunk boundary.

    Measures spectral difference between the last context_frames of chunk_a
    and the first context_frames of chunk_b.

    LSD = sqrt(mean_over_freq_bins( (log10(P_a) - log10(P_b))^2 ))
    where P_a, P_b are power spectra averaged over context frames.

    Parameters
    ----------
    chunk_a : np.ndarray
        Audio samples for the first chunk (mono, float32).
    chunk_b : np.ndarray
        Audio samples for the second chunk (mono, float32).
    sr : int
        Sample rate in Hz (default: 48000 for ACE-Step output).
    n_fft : int
        STFT window size (default: 2048).
    hop_length : int
        STFT hop length (default: 512).
    context_frames : int
        Number of frames from each side of the boundary to average (default: 5).

    Returns
    -------
    float
        LSD value in dB (non-negative).
    """
    import librosa

    S_a = np.abs(librosa.stft(chunk_a, n_fft=n_fft, hop_length=hop_length)) ** 2
    S_b = np.abs(librosa.stft(chunk_b, n_fft=n_fft, hop_length=hop_length)) ** 2

    # Boundary context: last N frames of chunk_a, first N frames of chunk_b
    P_a = S_a[:, -context_frames:].mean(axis=1)
    P_b = S_b[:, :context_frames].mean(axis=1)

    eps = 1e-10
    lsd = np.sqrt(np.mean((10 * np.log10(P_a + eps) - 10 * np.log10(P_b + eps)) ** 2))
    return float(lsd)


def apply_crossfade(
    chunk_a: np.ndarray,
    chunk_b: np.ndarray,
    sr: int = 48000,
    fade_duration_s: float = CROSSFADE_DURATION_S,
) -> tuple:
    """Apply overlap-add crossfade between chunk_a and chunk_b.

    The overlap-add technique mixes a fade-out of chunk_a with a fade-in of
    chunk_b in the boundary region. The returned chunks share a blended
    overlap zone: chunk_a_out ends with a mix of both signals fading out A
    and fading in B; chunk_b_out begins with the complementary blend.

    This guarantees that LSD measured at the boundary (last N frames of
    chunk_a_out vs first N frames of chunk_b_out) sees spectrally similar
    content on both sides -- the blend converges toward the same mixture --
    which is why crossfaded LSD is lower than hard-cut LSD.

    Parameters
    ----------
    chunk_a : np.ndarray
        First audio chunk (mono, float32).
    chunk_b : np.ndarray
        Second audio chunk (mono, float32).
    sr : int
        Sample rate in Hz (default: 48000).
    fade_duration_s : float
        Duration of the overlap zone in seconds (default: 0.5).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (chunk_a_out, chunk_b_out) -- chunk_a_out has a blended tail,
        chunk_b_out has the complementary blended head.
    """
    fade_samples = int(fade_duration_s * sr)
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)

    chunk_a_out = chunk_a.copy()
    chunk_b_out = chunk_b.copy()

    len_b_overlap = min(fade_samples, len(chunk_b_out))
    len_a_overlap = min(fade_samples, len(chunk_a_out))

    # chunk_a_out tail: fade_out(chunk_a) + fade_in(chunk_b) -- blended zone ending with B
    chunk_a_out[-len_a_overlap:] = (
        chunk_a[-len_a_overlap:] * fade_out[-len_a_overlap:]
        + chunk_b[:len_a_overlap] * fade_in[:len_a_overlap]
    )
    # chunk_b_out head: fade_out(chunk_a) + fade_in(chunk_b) -- blended zone starting with B
    chunk_b_out[:len_b_overlap] = (
        chunk_a[-len_b_overlap:] * fade_out[-len_b_overlap:]
        + chunk_b[:len_b_overlap] * fade_in[:len_b_overlap]
    )
    return chunk_a_out, chunk_b_out


def generate_server_chunks(
    adapter_name: str,
    n_chunks: int = N_CHUNKS,
    tmp_dir: str = None,
    server_url: str = SERVER_URL,
) -> tuple:
    """Generate n_chunks sequential WAVs via /generate and return their paths.

    Adapter identity is verified by calling /health before the loop and logging
    lora_path. sys.exit(1) if server is unreachable.

    Parameters
    ----------
    adapter_name : str
        Adapter name tag (for logging; actual adapter loaded on server).
    n_chunks : int
        Number of sequential /generate calls (default: 15).
    tmp_dir : str or None
        Directory to save chunk WAVs. If None, uses tempfile.mkdtemp().
    server_url : str
        Base URL of the AffectScore server.

    Returns
    -------
    tuple[list[str], str]
        (chunk_paths, lora_path) -- ordered WAV file paths and the adapter
        identity string read from /health for result JSON traceability.
    """
    import requests

    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="tc_chunks_")

    print(f"[AffectScore] Checking server health at {server_url}/health ...")
    try:
        health_resp = requests.get(f"{server_url}/health", timeout=10)
        health_resp.raise_for_status()
        health_data = health_resp.json()
        lora_path = health_data.get("lora_path", "unknown")
        print(f"[AffectScore] Server status: {health_data.get('status')}")
        print(f"[AffectScore] Adapter (lora_path): {lora_path}")
        if lora_path in ("none", None, "null", ""):
            print(f"[AffectScore] WARNING: Server reports no adapter loaded "
                  f"(lora_path={lora_path!r}). Expected adapter '{adapter_name}'.")
    except Exception as exc:
        print(f"[AffectScore] ERROR: Server not reachable at {server_url}: {exc}")
        sys.exit(1)

    # Fixed conditioning for all 15 calls -- same scene, simulates real deployment
    payload = {
        "affect_embedding": [0.0] * 512,
        "style_prompt": "neutral ambient orchestral",
        "valence": 0.0,
        "arousal": 0.0,
        "chunk_duration_s": 4.0,
        "max_latency_ms": 30000,
    }

    chunk_paths = []
    print(f"[AffectScore] Generating {n_chunks} sequential chunks (adapter: {adapter_name}) ...")

    for i in range(n_chunks):
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{server_url}/generate", json=payload, timeout=60)
            resp.raise_for_status()
        except Exception as exc:
            print(f"[AffectScore] Chunk {i + 1}/{n_chunks} FAILED: {exc}")
            sys.exit(1)

        chunk_path = os.path.join(tmp_dir, f"chunk_{i:02d}.wav")
        with open(chunk_path, "wb") as f:
            f.write(resp.content)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[AffectScore] Chunk {i + 1}/{n_chunks} generated ({elapsed_ms:.0f}ms)")
        chunk_paths.append(chunk_path)

    return chunk_paths, lora_path


def compute_natural_baseline(
    held_out_path: str,
    sr: int = SR_LOAD,
    preprocessed_dir: str = None,
) -> list:
    """Compute LSD at natural within-track boundaries for 200 held-out clips.

    For each clip, split at 25%, 50%, 75% of clip duration.
    Returns approximately 600 LSD values as the Wilcoxon baseline.

    Parameters
    ----------
    held_out_path : str
        Path to held_out_set.json.
    sr : int
        Sample rate to load at (default: 48000).
    preprocessed_dir : str, optional
        Directory containing preprocessed WAV files. If provided, searched first
        before the default data/preprocessed/ location.

    Returns
    -------
    list of float
        LSD values from natural within-track splits.
    """
    import librosa

    print(f"[AffectScore] Computing natural baseline from {held_out_path} ...")

    held_out_dir = os.path.dirname(os.path.abspath(held_out_path))

    with open(held_out_path) as f:
        clips = json.load(f)

    clips = clips[:N_BASELINE_CLIPS]
    natural_lsd_values = []
    skipped = 0

    for i, clip in enumerate(clips):
        clip_id = clip.get("clip_id", f"clip_{i}")
        filename = clip.get("filename", f"{clip_id}.mp3")

        candidates = []
        if preprocessed_dir:
            candidates += [
                os.path.join(preprocessed_dir, f"{clip_id}.wav"),
                os.path.join(preprocessed_dir, filename.replace(".mp3", ".wav")),
            ]
        candidates += [
            os.path.join(_REPO_ROOT, "data", "preprocessed", f"{clip_id}.wav"),
            os.path.join(_REPO_ROOT, "data", "preprocessed", filename.replace(".mp3", ".wav")),
            os.path.join(held_out_dir, "preprocessed", f"{clip_id}.wav"),
        ]
        wav_path = None
        for cand in candidates:
            if os.path.exists(cand):
                wav_path = cand
                break

        if wav_path is None:
            skipped += 1
            continue

        try:
            y, _ = librosa.load(wav_path, sr=sr, mono=True)
        except Exception as exc:
            print(f"[AffectScore] Skipping {clip_id} (load error): {exc}")
            skipped += 1
            continue

        dur = len(y)
        for frac in [0.25, 0.50, 0.75]:
            split = int(dur * frac)
            if split < LSD_CONTEXT_FRAMES * LSD_HOP_LENGTH + LSD_N_FFT:
                continue
            lsd = compute_lsd_at_boundary(y[:split], y[split:], sr=sr)
            natural_lsd_values.append(lsd)

        if (i + 1) % 50 == 0:
            print(f"[AffectScore] Natural baseline: {i + 1}/{len(clips)} clips "
                  f"({skipped} skipped, {len(natural_lsd_values)} values so far)")

    print(f"[AffectScore] Natural baseline complete: "
          f"{len(natural_lsd_values)} values from {len(clips) - skipped} clips "
          f"({skipped} skipped)")

    return natural_lsd_values


def save_spectrogram_figure(
    chunk_paths: list,
    condition_name: str,
    output_dir: str,
    chunk_duration_s: float = 4.0,
) -> None:
    """Save a 4-chunk spectrogram strip (PDF + PNG) for the given condition.

    Parameters
    ----------
    chunk_paths : list of str
        Paths to the first 4 chunk WAVs (only first 4 used).
    condition_name : str
        Condition label, e.g. "hard_cut" or "crossfaded".
    output_dir : str
        Directory where figures are saved.
    chunk_duration_s : float
        Duration of each chunk in seconds (for boundary marker placement).
    """
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    n_show = min(4, len(chunk_paths))
    fig, axes = plt.subplots(n_show, 1, figsize=(12, 8), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, chunk_path in enumerate(chunk_paths[:n_show]):
        y, sr_loaded = librosa.load(chunk_path, sr=SR_SPEC, mono=True)
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(
            S,
            ax=axes[i],
            sr=sr_loaded,
            hop_length=512,
            x_axis="time",
            y_axis="mel",
        )
        axes[i].set_title(f"Chunk {i + 1}", fontsize=9)
        axes[i].axvline(
            x=chunk_duration_s,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="boundary" if i == 0 else None,
        )
        if i == 0:
            axes[i].legend(fontsize=8)

    cond_label = condition_name.replace("_", " ").title()
    fig.suptitle(f"AffectScore Temporal Coherence -- {cond_label}", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    base_name = os.path.join(output_dir, f"temporal_coherence_{condition_name}")
    plt.savefig(base_name + ".pdf", dpi=150, bbox_inches="tight")
    plt.savefig(base_name + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[AffectScore] Figure saved: {base_name}.pdf / .png")


def run_temporal_coherence(
    adapter_name: str,
    held_out_path: str,
    server_url: str = SERVER_URL,
    preprocessed_dir: str = None,
    n_chunks: int = N_CHUNKS,
) -> dict:
    """Run the full temporal coherence evaluation.

    Generates n_chunks sequential chunks via server-loop, computes LSD at all
    boundaries in hard-cut and crossfaded conditions, runs Wilcoxon rank-sum
    tests against a natural within-track baseline, and saves results + figures.

    Parameters
    ----------
    adapter_name : str
        Name tag for the adapter under evaluation (e.g., "full", "no-affect").
    held_out_path : str
        Path to data/held_out_set.json.
    server_url : str
        Base URL of the AffectScore server.
    n_chunks : int
        Number of sequential /generate calls (default 15, giving 14 boundaries).
        Increase for more boundaries and better statistical power.

    Returns
    -------
    dict
        Full result dictionary (also saved to JSON).
    """
    import librosa
    from scipy.stats import ranksums

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DOCS_FIGURES, exist_ok=True)

    n_boundaries = n_chunks - 1
    print(f"[AffectScore] Generating {n_chunks} chunks -> {n_boundaries} boundaries.")
    chunk_paths, lora_path = generate_server_chunks(
        adapter_name=adapter_name,
        n_chunks=n_chunks,
        server_url=server_url,
    )

    print(f"[AffectScore] Loading {len(chunk_paths)} chunks at {SR_LOAD}Hz ...")
    chunks = []
    for path in chunk_paths:
        y, _ = librosa.load(path, sr=SR_LOAD, mono=True)
        chunks.append(y)

    print(f"[AffectScore] Computing hard-cut LSD at {n_boundaries} boundaries ...")
    hard_cut_lsd = []
    for i in range(len(chunks) - 1):
        lsd = compute_lsd_at_boundary(chunks[i], chunks[i + 1], sr=SR_LOAD)
        hard_cut_lsd.append(lsd)
        print(f"[AffectScore]   Boundary {i + 1:2d}/{len(chunks) - 1}: LSD={lsd:.4f} dB")

    print(f"[AffectScore] Computing crossfaded LSD at {n_boundaries} boundaries ...")
    crossfaded_lsd = []
    for i in range(len(chunks) - 1):
        a_faded, b_faded = apply_crossfade(chunks[i], chunks[i + 1], sr=SR_LOAD)
        lsd = compute_lsd_at_boundary(a_faded, b_faded, sr=SR_LOAD)
        crossfaded_lsd.append(lsd)
        print(f"[AffectScore]   Boundary {i + 1:2d}/{len(chunks) - 1}: "
              f"LSD={lsd:.4f} dB (crossfaded)")

    natural_lsd = compute_natural_baseline(held_out_path, sr=SR_LOAD,
                                            preprocessed_dir=preprocessed_dir)

    if not natural_lsd:
        raise RuntimeError(
            "Natural baseline is empty. Preprocessed WAVs must be available at "
            "data/preprocessed/ before running temporal coherence eval. Cannot proceed with "
            "Wilcoxon test against a fabricated baseline."
        )

    print("[AffectScore] Running Wilcoxon rank-sum tests ...")
    hc_stat, hc_p = ranksums(hard_cut_lsd, natural_lsd)
    cf_stat, cf_p = ranksums(crossfaded_lsd, natural_lsd)

    print(f"[AffectScore] Hard-cut vs natural:    stat={hc_stat:.4f}  p={hc_p:.4f}")
    print(f"[AffectScore] Crossfaded vs natural:  stat={cf_stat:.4f}  p={cf_p:.4f}")

    save_spectrogram_figure(
        chunk_paths=chunk_paths,
        condition_name="hard_cut",
        output_dir=DOCS_FIGURES,
    )

    import soundfile as sf
    tmp_cf_dir = tempfile.mkdtemp(prefix="tc_crossfaded_")
    for i in range(min(4, len(chunks))):
        if i < len(chunks) - 1:
            a_faded, _ = apply_crossfade(chunks[i], chunks[i + 1], sr=SR_LOAD)
        else:
            a_faded = chunks[i]
        sf.write(os.path.join(tmp_cf_dir, f"chunk_{i:02d}.wav"), a_faded, SR_LOAD)

    cf_paths = [os.path.join(tmp_cf_dir, f"chunk_{i:02d}.wav") for i in range(min(4, len(chunks)))]
    save_spectrogram_figure(
        chunk_paths=cf_paths,
        condition_name="crossfaded",
        output_dir=DOCS_FIGURES,
    )

    results = {
        "adapter_name": adapter_name,
        "lora_path": lora_path,
        "n_chunks": n_chunks,
        "n_boundaries": n_boundaries,
        "hard_cut": {
            "lsd_mean": float(np.mean(hard_cut_lsd)),
            "lsd_std": float(np.std(hard_cut_lsd)),
            "lsd_values": [float(v) for v in hard_cut_lsd],
            "wilcoxon_stat": float(hc_stat),
            "wilcoxon_p": float(hc_p),
            "is_primary_measure": True,
        },
        "crossfaded": {
            "lsd_mean": float(np.mean(crossfaded_lsd)),
            "lsd_std": float(np.std(crossfaded_lsd)),
            "lsd_values": [float(v) for v in crossfaded_lsd],
            "wilcoxon_stat": float(cf_stat),
            "wilcoxon_p": float(cf_p),
        },
        "natural_baseline": {
            "lsd_mean": float(np.mean(natural_lsd)),
            "lsd_std": float(np.std(natural_lsd)),
            "n_measurements": len(natural_lsd),
        },
        "lsd_params": {
            "n_fft": LSD_N_FFT,
            "hop_length": LSD_HOP_LENGTH,
            "context_frames": LSD_CONTEXT_FRAMES,
        },
    }

    results_path = os.path.join(RESULTS_DIR, "temporal_coherence_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[AffectScore] Results saved to {results_path}")

    print(f"\n[AffectScore] === Temporal Coherence Summary ===")
    print(f"  Adapter:            {adapter_name}  (lora_path={lora_path})")
    print(f"  Hard-cut LSD:       mean={results['hard_cut']['lsd_mean']:.4f} dB  "
          f"std={results['hard_cut']['lsd_std']:.4f}  [PRIMARY MEASURE]")
    print(f"  Crossfaded LSD:     mean={results['crossfaded']['lsd_mean']:.4f} dB  "
          f"std={results['crossfaded']['lsd_std']:.4f}")
    print(f"  Natural baseline:   mean={results['natural_baseline']['lsd_mean']:.4f} dB  "
          f"n={results['natural_baseline']['n_measurements']}")
    print(f"  Wilcoxon hard-cut:  stat={hc_stat:.4f}  p={hc_p:.4f}")
    print(f"  Wilcoxon crossfade: stat={cf_stat:.4f}  p={cf_p:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Temporal coherence evaluation via 15-chunk server-loop"
    )
    parser.add_argument(
        "--adapter-name",
        type=str,
        required=True,
        help="Adapter name tag (e.g., 'full', 'no-affect', 'r32'). "
             "The actual adapter is loaded on the server via --lora arg at server start.",
    )
    parser.add_argument(
        "--held-out",
        type=str,
        default=os.path.join(_REPO_ROOT, "data", "held_out_set.json"),
        help="Path to held_out_set.json (default: data/held_out_set.json)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=SERVER_URL,
        help=f"AffectScore server URL (default: {SERVER_URL})",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default=None,
        help="Directory of preprocessed WAV files for natural baseline "
             "(e.g. /content/drive/MyDrive/affectscore/preprocessed_unfiltered). "
             "Falls back to data/preprocessed/ if omitted.",
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=N_CHUNKS,
        help=(
            f"Number of sequential /generate calls (default: {N_CHUNKS}, giving "
            f"{N_CHUNKS - 1} boundaries). Increase for better Wilcoxon power."
        ),
    )
    args = parser.parse_args()

    run_temporal_coherence(
        adapter_name=args.adapter_name,
        held_out_path=args.held_out,
        server_url=args.server_url,
        preprocessed_dir=args.preprocessed_dir,
        n_chunks=args.n_chunks,
    )
