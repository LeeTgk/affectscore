"""
Computational probe of behavioral response space.
Validates that different player engagement signals (same designer intent V=0/A=0,
different archetype behavior profiles) produce measurably different audio
distributions -- key evidence for the engagement-modulation layer (Layer 2).

Layer 2 carries player engagement signals (choice latency, dwell deviation,
interaction rate) -- NOT internal psychological states. Fixed designer intent
V=0.0/A=0.0 throughout. With n=12 clips per archetype, FAD estimates
will have high variance -- this is acknowledged as a "computational probe."

Usage (Colab A100):
    python eval/simulated_archetypes.py \\
        --archetype-base-dir /content/drive/MyDrive/affectscore/eval_outputs/full/ \\
        --adapter-name full
"""

import os
import sys
import json
import argparse
import itertools

# Multiprocessing patches -- same as audio_metrics.py.
# Python 3.12 + CUDA: fork-context SemLocks cannot cross spawn process boundary.
# fadtk and tqdm use Pool/ProcessPoolExecutor which crash silently in Colab A100.
# Patches run all tasks inline (no subprocess). Must run BEFORE any library import.
import multiprocessing as _mp
import multiprocessing.pool as _mp_pool
import concurrent.futures as _cf

class _InlinePool:
    class _AsyncResult:
        def __init__(self, v): self._v = v
        def get(self, timeout=None): return self._v
        def ready(self): return True
        def successful(self): return True
        def wait(self, timeout=None): pass
    def __init__(self, processes=None, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def map(self, fn, it, chunksize=None): return list(map(fn, it))
    def starmap(self, fn, it, chunksize=None): return [fn(*a) for a in it]
    def imap(self, fn, it, chunksize=1): return map(fn, it)
    def imap_unordered(self, fn, it, chunksize=1): return map(fn, it)
    def apply(self, fn, args=(), kwds={}): return fn(*args, **kwds)
    def apply_async(self, fn, args=(), kwds={}, callback=None, error_callback=None):
        try:
            r = fn(*args, **kwds)
            if callback: callback(r)
            return _InlinePool._AsyncResult(r)
        except Exception as e:
            if error_callback: error_callback(e)
            raise
    def close(self): pass
    def join(self): pass
    def terminate(self): pass

class _InlineFuture:
    def __init__(self, result=None, exception=None):
        self._result = result
        self._exception = exception
    def result(self, timeout=None):
        if self._exception: raise self._exception
        return self._result
    def exception(self, timeout=None): return self._exception
    def done(self): return True
    def cancelled(self): return False
    def running(self): return False
    def cancel(self): return False
    def add_done_callback(self, fn): fn(self)

class _InlineExecutor:
    def __init__(self, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def submit(self, fn, /, *args, **kwargs):
        try:
            return _InlineFuture(result=fn(*args, **kwargs))
        except Exception as e:
            return _InlineFuture(exception=e)
    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return (fn(*args) for args in zip(*iterables))
    def shutdown(self, wait=True, **kwargs): pass

_mp.Pool = _InlinePool
_mp_pool.Pool = _InlinePool
_cf.ProcessPoolExecutor = _InlineExecutor

import torch.utils.data as _tud
_orig_dl_init = _tud.DataLoader.__init__
def _dl_init_no_workers(self, *args, **kwargs):
    kwargs['num_workers'] = 0
    for _k in ('prefetch_factor', 'persistent_workers', 'multiprocessing_context'):
        kwargs.pop(_k, None)
    _orig_dl_init(self, *args, **kwargs)
_tud.DataLoader.__init__ = _dl_init_no_workers

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
from scipy.linalg import sqrtm

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_HERE, "results")
DOCS_FIGURES = os.path.join(_REPO_ROOT, "docs", "figures")
MERT_MODEL_NAME = "MERT-v1-95M-6"

# These engagement signal definitions are shared with generate_eval_set.py.

#: Four archetype engagement signal profiles.
#: All profiles share V=0.0/A=0.0 designer intent -- engagement signals only.
ARCHETYPES = {
    "contemplative": {
        "arc_position":           0.3,
        "choice_latency_norm":    0.8,
        "dwell_deviation_norm":   0.7,
        "interaction_rate_norm":  0.2,
    },
    "impulsive": {
        "arc_position":           0.3,
        "choice_latency_norm":    0.1,
        "dwell_deviation_norm":   0.2,
        "interaction_rate_norm":  0.9,
    },
    "tense": {
        "arc_position":           0.7,
        "choice_latency_norm":    0.6,
        "dwell_deviation_norm":   0.3,
        "interaction_rate_norm":  0.5,
    },
    "neutral": {
        "arc_position":           0.5,
        "choice_latency_norm":    0.5,
        "dwell_deviation_norm":   0.5,
        "interaction_rate_norm":  0.5,
    },
}

#: Controlled archetype profiles -- arc_position=0.5 for all archetypes.
#: Only Layer-2 engagement signals differ across archetypes; arc_position is held constant.
#: Used with --controlled flag on both generate_eval_set.py and this script.
ARCHETYPES_CONTROLLED = {
    "contemplative": {
        "arc_position":           0.5,
        "choice_latency_norm":    0.8,
        "dwell_deviation_norm":   0.7,
        "interaction_rate_norm":  0.2,
    },
    "impulsive": {
        "arc_position":           0.5,
        "choice_latency_norm":    0.1,
        "dwell_deviation_norm":   0.2,
        "interaction_rate_norm":  0.9,
    },
    "tense": {
        "arc_position":           0.5,
        "choice_latency_norm":    0.6,
        "dwell_deviation_norm":   0.3,
        "interaction_rate_norm":  0.5,
    },
    "neutral": {
        "arc_position":           0.5,
        "choice_latency_norm":    0.5,
        "dwell_deviation_norm":   0.5,
        "interaction_rate_norm":  0.5,
    },
}

#: Fixed designer-intent anchor (V=0.0, A=0.0).
DESIGNER_VALENCE = 0.0
DESIGNER_AROUSAL = 0.0

#: Style prompt used for all archetype clip generation.
STYLE_PROMPT = "ambient orchestral game soundtrack neutral calm"

#: Canonical archetype ordering for matrix rows/columns.
ARCHETYPE_NAMES = ["contemplative", "impulsive", "tense", "neutral"]

#: Number of generated clips per archetype.
N_ARCHETYPE_CLIPS = 12

#: Bonferroni correction over C(4,2) = 6 unique pairs.
BONFERRONI_N_PAIRS = 6
ALPHA_CORRECTED = 0.05 / BONFERRONI_N_PAIRS   # approx 0.0083


def frechet_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Frechet distance between two sets of embeddings.

    Uses scipy.linalg.sqrtm for numerical stability; .real discards small
    imaginary artifacts from floating-point matrix ops.

    Args:
        X: (n_a, D) array of embeddings for group A.
        Y: (n_b, D) array of embeddings for group B.

    Returns:
        Scalar Frechet distance (float). Always >= 0.
    """
    mu_x = X.mean(0)
    sigma_x = np.cov(X, rowvar=False)
    mu_y = Y.mean(0)
    sigma_y = np.cov(Y, rowvar=False)

    diff = mu_x - mu_y
    covmean = sqrtm(sigma_x @ sigma_y).real   # .real strips numerical imaginary artifacts
    return float(np.dot(diff, diff) + np.trace(sigma_x + sigma_y - 2 * covmean))


def permutation_fad_pvalue(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    n_permutations: int = 1000,
) -> float:
    """One-tailed permutation p-value for FAD(A, B).

    This is an embedding-space permutation test: MERT embeddings are extracted
    once and group labels are permuted in-place. This avoids re-running the audio
    encoder 1000 times per pair (which would take hours on A100 for 24 clips x 6 pairs).

    Args:
        emb_a: (n_a, D) embeddings for archetype A.
        emb_b: (n_b, D) embeddings for archetype B.
        n_permutations: Number of permutation trials (default 1000).

    Returns:
        p-value in [0, 1]. Includes +1 continuity correction so the
        minimum achievable p-value is 1/(n_permutations+1) approx 0.001.
    """
    observed_fad = frechet_distance(emb_a, emb_b)
    pooled = np.vstack([emb_a, emb_b])
    n_a = len(emb_a)
    n_more_extreme = 0
    for _ in range(n_permutations):
        perm = np.random.permutation(len(pooled))
        perm_fad = frechet_distance(pooled[perm[:n_a]], pooled[perm[n_a:]])
        if perm_fad >= observed_fad:
            n_more_extreme += 1
    return (n_more_extreme + 1) / (n_permutations + 1)   # continuity correction


def get_archetype_embeddings(archetype_dirs: dict, model) -> dict:
    """Extract MERT embeddings for each archetype directory via fadtk.

    Caches embeddings to disk so the MERT encoder runs only once per archetype
    (not once per permutation iteration).

    fadtk caches embeddings adjacent to each audio directory:
        .{dir_name}_{model.name}_emb/
    e.g. for /drive/eval_outputs/full/contemplative/:
        /drive/eval_outputs/full/.contemplative_MERT-v1-95M-6_emb/

    Args:
        archetype_dirs: {"contemplative": "/path/to/contemplative/", ...}
        model: fadtk model object (e.g. from get_all_models()["MERT-v1-95M-6"])

    Returns:
        {"contemplative": np.ndarray of shape (n, D), ...}
    """
    import shutil
    from fadtk.fad_batch import cache_embedding_files
    from pathlib import Path

    embeddings = {}
    for name, audio_dir in archetype_dirs.items():
        audio_path = Path(audio_dir)
        emb_dir = audio_path.parent / f".{audio_path.name}_{model.name}_emb"

        # Force-delete stale emb_dir so fadtk cannot false-positive "already cached"
        # (caused by subprocess crashes in Python 3.12 + CUDA leaving a partial or
        # mislocated cache). The _InlinePool patch above ensures .npy files are
        # written to emb_dir after this point.
        if emb_dir.exists():
            shutil.rmtree(emb_dir)
            print(f"[AffectScore] Cleared stale cache: {emb_dir}")

        print(f"[AffectScore] Caching MERT embeddings for archetype '{name}': {audio_path}")
        cache_embedding_files(audio_path, model, workers=0)

        if not emb_dir.exists():
            print(
                f"[AffectScore] WARNING: Embedding cache not found at expected path "
                f"{emb_dir}. Attempting fallback glob for *.npy files in parent..."
            )
            candidates = list(audio_path.parent.glob(f"*{model.name}*emb*"))
            if candidates:
                emb_dir = candidates[0]
                print(f"[AffectScore] Fallback cache dir: {emb_dir}")
            else:
                print(
                    f"[AffectScore] ERROR: No embedding cache found for '{name}'. "
                    f"Skipping -- this archetype will be excluded from FAD matrix."
                )
                continue

        npy_files = sorted(emb_dir.glob("*.npy"))
        if not npy_files:
            print(
                f"[AffectScore] WARNING: No .npy files in {emb_dir} for '{name}'. "
                f"Skipping archetype."
            )
            continue

        arrs = [np.load(str(f)) for f in npy_files]
        embeddings[name] = np.stack(arrs)
        print(f"[AffectScore] Loaded {len(arrs)} embeddings for '{name}' -- shape {embeddings[name].shape}")

    return embeddings


def compute_fad_matrix(embeddings: dict, n_permutations: int = 1000) -> dict:
    """Compute 4x4 pairwise FAD matrix and permutation p-values.

    Args:
        embeddings: {"contemplative": (n, D), ...} from get_archetype_embeddings.
        n_permutations: Permutation count for each pair's p-value test.

    Returns:
        Dict with keys: archetype_names, fad_matrix, pairwise, alpha_corrected,
        n_permutations.
    """
    names = [n for n in ARCHETYPE_NAMES if n in embeddings]

    full_names = ARCHETYPE_NAMES
    full_n = len(full_names)
    fad_mat = [[0.0] * full_n for _ in range(full_n)]
    pairwise = {}

    for name_a, name_b in itertools.combinations(names, 2):
        i = full_names.index(name_a)
        j = full_names.index(name_b)
        pair_key = f"{name_a}_vs_{name_b}"

        print(f"[AffectScore] Computing FAD: {name_a} vs {name_b} ...")
        emb_a = embeddings[name_a]
        emb_b = embeddings[name_b]

        fad_val = frechet_distance(emb_a, emb_b)
        fad_mat[i][j] = fad_val
        fad_mat[j][i] = fad_val

        print(f"[AffectScore]   FAD({name_a}, {name_b}) = {fad_val:.4f}")
        print(f"[AffectScore]   Running permutation test ({n_permutations} permutations)...")
        p_raw = permutation_fad_pvalue(emb_a, emb_b, n_permutations=n_permutations)
        p_corrected = min(1.0, p_raw * BONFERRONI_N_PAIRS)
        significant = p_corrected < ALPHA_CORRECTED

        print(f"[AffectScore]   p_raw={p_raw:.4f}, p_corrected={p_corrected:.4f}, "
              f"significant={significant}")

        pairwise[pair_key] = {
            "fad":         fad_val,
            "p_raw":       p_raw,
            "p_corrected": p_corrected,
            "significant": significant,
        }

    return {
        "archetype_names": full_names,
        "fad_matrix":      fad_mat,
        "pairwise":        pairwise,
        "alpha_corrected": ALPHA_CORRECTED,
        "n_permutations":  n_permutations,
    }


def save_spectrogram_strip(archetype_dirs: dict, output_dir: str) -> None:
    """Save a 4-panel spectrogram strip (one representative clip per archetype).

    Output is a vertically-stacked figure suitable for the manuscript's
    Appendix / Computational Results section.

    Args:
        archetype_dirs: {"contemplative": "/path/to/dir/", ...}
        output_dir: Directory to write archetype_spectrograms.{pdf,png}.
    """
    import librosa.display

    os.makedirs(output_dir, exist_ok=True)

    representative_clips = []
    for name in ARCHETYPE_NAMES:
        if name not in archetype_dirs:
            representative_clips.append((name, None))
            continue
        wav_files = sorted(
            f for f in os.listdir(archetype_dirs[name])
            if f.lower().endswith(".wav")
        )
        if wav_files:
            representative_clips.append(
                (name, os.path.join(archetype_dirs[name], wav_files[0]))
            )
        else:
            representative_clips.append((name, None))

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    for i, (name, wav_path) in enumerate(representative_clips):
        ax = axes[i]
        if wav_path is None or not os.path.exists(wav_path):
            ax.text(0.5, 0.5, f"No WAV found for '{name}'",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Archetype: {name}")
            continue
        y, sr = librosa.load(wav_path, sr=22050)
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(S, ax=ax, sr=sr, x_axis="time", y_axis="mel")
        ax.set_title(f"Archetype: {name}")

    plt.suptitle(
        "Per-archetype spectrogram strip (representative clip)\n"
        "Designer intent: V=0.0, A=0.0 (neutral) for all archetypes",
        fontsize=10,
    )
    plt.tight_layout()

    pdf_path = os.path.join(output_dir, "archetype_spectrograms.pdf")
    png_path = os.path.join(output_dir, "archetype_spectrograms.png")
    plt.savefig(pdf_path, bbox_inches="tight", dpi=150)
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"[AffectScore] Spectrogram strip saved:")
    print(f"[AffectScore]   {pdf_path}")
    print(f"[AffectScore]   {png_path}")


def run_simulated_archetypes(
    archetype_base_dir: str,
    adapter_name: str,
    n_permutations: int = 1000,
    controlled: bool = False,
) -> dict:
    """Run the full simulated archetypes evaluation.

    Orchestrates MERT embedding extraction, 4x4 pairwise FAD matrix,
    embedding-space permutation tests (1000 per pair), Bonferroni correction
    over 6 unique pairs, result JSON, and spectrogram strip figure.

    Args:
        archetype_base_dir: Directory containing contemplative/, impulsive/,
            tense/, neutral/ subdirectories. When controlled=True, point to
            output from generate_eval_set.py --controlled.
        adapter_name: Adapter variant name -- used in result file naming.
        n_permutations: Permutation count (default 1000).
        controlled: If True, records ARCHETYPES_CONTROLLED in results JSON.
            Result filename is simulated_archetypes_results_{adapter}_ctrl.json.

    Returns:
        Results dict (also saved as JSON).
    """
    ctrl_label = "_ctrl" if controlled else ""
    print(f"[AffectScore] === Simulated Archetypes (adapter={adapter_name}{ctrl_label}) ===")
    print(f"[AffectScore] Archetype base dir: {archetype_base_dir}")
    print(f"[AffectScore] Designer anchor: V={DESIGNER_VALENCE}, A={DESIGNER_AROUSAL}")
    if controlled:
        print(f"[AffectScore] Controlled mode: arc_position=0.5 for all archetypes.")
        print(f"[AffectScore]   Clips must have been generated with generate_eval_set.py --controlled.")
    print(f"[AffectScore] n_permutations={n_permutations}, "
          f"Bonferroni n_pairs={BONFERRONI_N_PAIRS}, "
          f"alpha_corrected={ALPHA_CORRECTED:.4f}")

    archetype_dirs = {}
    for name in ARCHETYPE_NAMES:
        d = os.path.join(archetype_base_dir, name)
        if os.path.isdir(d):
            archetype_dirs[name] = d
        else:
            print(f"[AffectScore] WARNING: Archetype directory not found: {d}")

    if len(archetype_dirs) < 2:
        print(f"[AffectScore] ERROR: Need at least 2 archetype directories, "
              f"found {len(archetype_dirs)}.")
        return {"error": "Insufficient archetype directories", "found": list(archetype_dirs)}

    print(f"[AffectScore] Loading fadtk model: {MERT_MODEL_NAME}")
    from fadtk.model_loader import get_all_models
    models = {m.name: m for m in get_all_models()}
    if MERT_MODEL_NAME not in models:
        available = list(models.keys())
        raise ValueError(
            f"MERT model '{MERT_MODEL_NAME}' not found in fadtk. "
            f"Available: {available[:10]}..."
        )
    mert_model = models[MERT_MODEL_NAME]
    print(f"[AffectScore] Model loaded: {mert_model.name}")

    embeddings = get_archetype_embeddings(archetype_dirs, mert_model)

    if len(embeddings) < 2:
        print(f"[AffectScore] ERROR: Embedding extraction failed for most archetypes.")
        return {"error": "Embedding extraction failed", "loaded": list(embeddings)}

    matrix_results = compute_fad_matrix(embeddings, n_permutations=n_permutations)

    results = {
        "archetype_names":      matrix_results["archetype_names"],
        "fad_matrix":           matrix_results["fad_matrix"],
        "pairwise":             matrix_results["pairwise"],
        "alpha_corrected":      ALPHA_CORRECTED,
        "n_permutations":       n_permutations,
        "mert_model":           MERT_MODEL_NAME,
        "n_clips_per_archetype": N_ARCHETYPE_CLIPS,
        "designer_valence":     DESIGNER_VALENCE,
        "designer_arousal":     DESIGNER_AROUSAL,
        "adapter_name":         adapter_name,
        "controlled":           controlled,
        "arc_position_note": (
            "Controlled: all arc_position=0.5"
            if controlled else
            "Standard: tense arc_position=0.7, others <=0.5"
        ),
        "framing": (
            "Computational probe of behavioral response space. "
            "Layer 2 carries player engagement signals (choice latency, "
            "dwell deviation, interaction rate) -- NOT internal psychological states. "
            "Fixed designer intent V=0.0/A=0.0 throughout."
        ),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(
        RESULTS_DIR, f"simulated_archetypes_results_{adapter_name}{ctrl_label}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[AffectScore] Results saved to {results_path}")

    os.makedirs(DOCS_FIGURES, exist_ok=True)
    save_spectrogram_strip(archetype_dirs, DOCS_FIGURES)

    print(f"\n[AffectScore] === Pairwise FAD Matrix (MERT backbone) ===")
    header = "           " + "  ".join(f"{n[:10]:>10}" for n in ARCHETYPE_NAMES)
    print(header)
    for i, row_name in enumerate(ARCHETYPE_NAMES):
        row_vals = "  ".join(f"{matrix_results['fad_matrix'][i][j]:10.4f}"
                             for j in range(len(ARCHETYPE_NAMES)))
        print(f"  {row_name[:10]:<10} {row_vals}")

    print(f"\n[AffectScore] === Pairwise Significance (Bonferroni-corrected) ===")
    for pair_key, pair_data in matrix_results["pairwise"].items():
        sig_star = "*" if pair_data["significant"] else ""
        print(f"  {pair_key:40s}  FAD={pair_data['fad']:.4f}  "
              f"p_raw={pair_data['p_raw']:.4f}  "
              f"p_corr={pair_data['p_corrected']:.4f}  {sig_star}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Simulated Archetypes -- computational probe of behavioral "
            "response space. Computes 4x4 pairwise FAD matrix (MERT backbone) "
            "with Bonferroni-corrected permutation test p-values."
        )
    )
    parser.add_argument(
        "--archetype-base-dir",
        type=str,
        required=True,
        help=(
            "Base directory containing archetype subdirectories: "
            "contemplative/, impulsive/, tense/, neutral/. "
            "E.g. /content/drive/MyDrive/affectscore/eval_outputs/full/"
        ),
    )
    parser.add_argument(
        "--adapter-name",
        type=str,
        required=True,
        help="Adapter variant name (e.g. 'full', 'no-affect'). Used in output filenames.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutations per pair for p-value estimation (default: 1000).",
    )
    parser.add_argument(
        "--controlled",
        action="store_true",
        default=False,
        help=(
            "Record that clips were generated with ARCHETYPES_CONTROLLED (arc_position=0.5 "
            "for all archetypes). Requires archetype_base_dir to point at output from "
            "generate_eval_set.py --controlled. Result saved as "
            "simulated_archetypes_results_{adapter}_ctrl.json."
        ),
    )
    args = parser.parse_args()
    run_simulated_archetypes(
        archetype_base_dir=args.archetype_base_dir,
        adapter_name=args.adapter_name,
        n_permutations=args.n_permutations,
        controlled=args.controlled,
    )
