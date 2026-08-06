"""
Audio quality metrics for the AffectScore ablation table.
Computes FAD-MERT, CLAP-score, KLD, and PCE for each of the 6 adapter variants.

Usage:
    python eval/audio_metrics.py \\
        --audio-dir /path/to/full/ \\
        --reference-dir /path/to/preprocessed_unfiltered/ \\
        --adapter-name full \\
        --held-out data/held_out_set.json \\
        --reference-manifest data/training_set_clean_clap.json
"""

import os
import sys
import json
import argparse
import tempfile
import contextlib
import io

# No-subprocess patches -- MUST run before any library is imported
#
# Root cause: Python 3.12 + CUDA.  Linux default mp start-method is 'fork', so
# multiprocessing primitives (Queue, RLock, Semaphore) get _is_fork_ctx=True.
# PyTorch CUDA and concurrent.futures both create SpawnProcesses.
# Python 3.12 forbids sharing fork-context SemLocks with spawn processes.
#
# Full call-graph audit -- every subprocess-creating path in this script:
#
#   (1) fadtk.fad_batch.cache_embedding_files
#         -> multiprocessing.Pool(workers)  <- Pool created even when workers=1
#         Source: fadtk may use `from multiprocessing import Pool`
#                 OR `from multiprocessing.pool import Pool`
#
#   (2) msclap.CLAP.get_audio_embeddings
#         -> torch.utils.data.DataLoader(dataset, num_workers=N)
#
#   (3) fadtk.FrechetAudioDistance.score  (if embeddings not yet cached)
#         -> may call cache_embedding_files internally -> same Pool as (1)
#
#   (4) hear21passt.load_model / model(wave)
#         -> pure forward pass, no DataLoader; low risk, covered by patch B anyway
#
#   (5) fadtk statistics phase ("Calculating statistics")
#         -> hypy_utils.tqdm_utils.pmap
#         -> tqdm.contrib.concurrent.process_map
#         -> concurrent.futures.ProcessPoolExecutor(max_workers=12)
#         SEPARATE from multiprocessing.Pool -- patches A/B do NOT cover this.
#         tqdm creates a fork-context RLock and tries to send it to spawn
#         workers via pickle -> same SemLock crash as (1).
#
# Fix strategy -- four global patches applied before any lazy import:
#
#   A) multiprocessing.Pool  +  multiprocessing.pool.Pool  ->  _InlinePool
#      Covers (1) and (3) regardless of whether fadtk uses
#        `from multiprocessing import Pool` or `from multiprocessing.pool import Pool`
#      Applied BEFORE `import torch` so torch.multiprocessing (which does
#        `from multiprocessing import *`) inherits _InlinePool automatically.
#
#   B) torch.utils.data.DataLoader.__init__  ->  force num_workers=0
#      Covers (2) and (3); all data loading runs in the main thread.
#
#   C) concurrent.futures.ProcessPoolExecutor  ->  _InlineExecutor
#      Covers (5) -- hypy_utils.pmap -> process_map -> ProcessPoolExecutor.
#      _InlineExecutor never creates any process so the fork-context RLock
#      never gets pickled and never crosses a process boundary.
#
#   D) TOKENIZERS_PARALLELISM=false  -- suppresses HuggingFace tokenizer forks

import multiprocessing as _mp
import multiprocessing.pool as _mp_pool
import concurrent.futures as _cf

class _InlinePool:
    """multiprocessing.Pool drop-in -- runs every task in the calling process.

    No subprocess is created, so no SemLock crosses a process boundary.
    Implements the full Pool surface area used by fadtk so the monkey-patch
    is transparent regardless of which Pool methods fadtk calls.
    """
    class _AsyncResult:
        def __init__(self, v): self._v = v
        def get(self, timeout=None): return self._v
        def ready(self): return True
        def successful(self): return True
        def wait(self, timeout=None): pass

    def __init__(self, processes=None, *a, **kw): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def map(self, fn, it, chunksize=None):
        return list(map(fn, it))
    def starmap(self, fn, it, chunksize=None):
        return [fn(*a) for a in it]
    def imap(self, fn, it, chunksize=1):
        return map(fn, it)
    def imap_unordered(self, fn, it, chunksize=1):
        return map(fn, it)
    def apply(self, fn, args=(), kwds={}):
        return fn(*args, **kwds)
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
    """concurrent.futures.Future drop-in -- result already computed inline."""
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
    """concurrent.futures.ProcessPoolExecutor drop-in -- runs all tasks inline.

    tqdm.contrib.concurrent.process_map creates a fork-context RLock (for
    tqdm thread-safety) and passes it as `initargs` to ProcessPoolExecutor.
    ProcessPoolExecutor tries to pickle it into each SpawnProcess -> crash.
    Running tasks inline means no spawn, no pickle, no crash.
    """
    def __init__(self, *args, **kwargs): pass
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


# Patch A: Pool -- must happen BEFORE `import torch` (torch.multiprocessing
# does `from multiprocessing import *` at import time and would inherit Pool).
_mp.Pool = _InlinePool
_mp_pool.Pool = _InlinePool

# Patch C: ProcessPoolExecutor -- covers hypy_utils.pmap -> process_map path.
_cf.ProcessPoolExecutor = _InlineExecutor

# Patch B: DataLoader -- import torch after Pool is patched.
import torch.utils.data as _tud
_orig_dl_init = _tud.DataLoader.__init__
def _dl_init_no_workers(self, *args, **kwargs):
    kwargs['num_workers'] = 0
    # These kwargs are only legal when num_workers > 0; silently drop them.
    for _k in ('prefetch_factor', 'persistent_workers', 'multiprocessing_context'):
        kwargs.pop(_k, None)
    _orig_dl_init(self, *args, **kwargs)
_tud.DataLoader.__init__ = _dl_init_no_workers

# Patch D: HuggingFace tokenizer parallelism.
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
import librosa
from pathlib import Path
from scipy.stats import entropy as scipy_entropy
import soundfile as sf


class _Tee:
    """Write stdout to both the terminal and a log file simultaneously."""
    def __init__(self, log_path: str):
        self._file = open(log_path, "w", encoding="utf-8", buffering=1)
        self._orig = sys.stdout
        sys.stdout = self
    def write(self, data):
        self._orig.write(data)
        self._file.write(data)
    def flush(self):
        self._orig.flush()
        self._file.flush()
    def close(self):
        sys.stdout = self._orig
        self._file.close()

def _make_wav_only_dir(source_dir: Path) -> Path:
    """Return a directory containing only audio-file symlinks from source_dir.

    ACEStepPipeline writes _input_params.json sidecars alongside every WAV.
    fadtk scans all files in the directory and passes them to libsndfile,
    raising LibsndfileError on non-audio files.

    Google Drive FUSE (errno 95 EOPNOTSUPP) does not support symlinks, so the
    symlink directory is always created in /tmp/ -- local Colab fs supports
    symlinks even when their targets are on Drive. fadtk's .npy embedding
    cache is written to /tmp/ and is lost between Colab sessions; for eval
    dirs (200 short clips) the recompute cost is acceptable (~2-5 min).
    """
    import hashlib
    AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a'}
    dir_hash = hashlib.md5(str(source_dir.resolve()).encode()).hexdigest()[:8]
    wav_dir = Path(f"/tmp/_fad_audio_{dir_hash}")
    wav_dir.mkdir(exist_ok=True, parents=True)
    for f in source_dir.iterdir():
        if f.suffix.lower() in AUDIO_EXTS:
            link = wav_dir / f.name
            if not link.exists():
                link.symlink_to(f.resolve())
    return wav_dir


_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_HERE, "results")

MERT_MODEL_NAME = "MERT-v1-95M-6"   # Layer 6 -- middle layer balancing acoustic/semantic
# VGGish removed: harritaylor/torchvggish is archived on GitHub and caused hub
# download errors in headless Colab. FAD-MERT is the sole FAD metric.

# Alternative FAD backbones for MERT circularity check.
# MERT plays three roles (annotation, FAD, MER), creating representational circularity.
# An independent backbone validates that the two-tier structure (no-lora vs LoRA variants)
# is not an artifact of MERT feature alignment.
ALT_BACKBONE_MODELS = [
    "encodec-emb",        # EnCodec 75Hz codec encoder (acoustic backbone)
    "clap-laion-music",   # CLAP music encoder (different training data from MERT)
]

SAMPLE_RATE_CLAP = 44100             # msclap expects 44.1 kHz
SAMPLE_RATE_PASST = 32000            # hear21passt expects 32 kHz
PASST_MIN_SAMPLES = 32000 * 10      # Zero-pad to 10s minimum (PaSST trained on 10s clips)
EPS_KLD = 1e-6


def compute_pce(wav_path: str, sr: int = 22050) -> float:
    """Pitch Class Entropy via chroma_stft -> normalize -> Shannon entropy.

    PCE = Shannon entropy (bits) of the mean chroma activation vector.
    Range: [0, log2(12)] approx [0, 3.585] bits. Higher = greater tonal diversity.
    """
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)   # (12, T)
    chroma_mean = chroma.mean(axis=1)                    # (12,) mean per pitch class
    chroma_norm = chroma_mean / (chroma_mean.sum() + 1e-8)
    return float(scipy_entropy(chroma_norm, base=2))


def get_passt_probs(wav_path: str, model) -> np.ndarray:
    """Get PaSST AudioSet 527-class softmax probabilities from a WAV file.

    Loads at 32 kHz mono and zero-pads to at least 10 seconds (PaSST trained
    on 10-second AudioSet clips) before running inference.
    """
    import torch

    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE_PASST, mono=True)

    # Truncate first (reference corpus clips are full songs, 30s-2min), then
    # zero-pad short clips. Both operations target PASST_MIN_SAMPLES (320,000).
    y = y[:PASST_MIN_SAMPLES]
    if len(y) < PASST_MIN_SAMPLES:
        y = np.pad(y, (0, PASST_MIN_SAMPLES - len(y)))

    wave = torch.from_numpy(y).unsqueeze(0).cuda()  # (1, n_samples)

    # hear21passt prints softmax arrays to stdout during inference.
    # Suppress with redirect_stdout/stderr -- we only need the return value.
    _sink = io.StringIO()
    with torch.no_grad(), contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
        logits = model(wave)   # (1, 527)

    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # (527,)
    return probs


def compute_clap_score(wav_path: str, style_prompt: str, clap_model) -> float:
    """Compute CLAP-score (text-audio cosine similarity) for one clip.

    Pre-resamples WAV from 48 kHz -> 44.1 kHz before calling get_audio_embeddings
    to ensure msclap receives its expected sample rate.
    """
    y_44k, _ = librosa.load(wav_path, sr=SAMPLE_RATE_CLAP, mono=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, y_44k, SAMPLE_RATE_CLAP)

    try:
        audio_emb = clap_model.get_audio_embeddings([tmp_path])
        text_emb = clap_model.get_text_embeddings([style_prompt])
        similarity = clap_model.compute_similarity(audio_emb, text_emb)
        return float(similarity[0][0])
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def compute_fad_for_pair(reference_dir: str, eval_dir: str, model_name: str) -> float:
    """Compute Frechet Audio Distance between two directories using fadtk.

    reference_dir and eval_dir must be SEPARATE directories per adapter variant.
    fadtk caches embeddings keyed by directory, not file content -- mixing
    directories produces stale cache.
    """
    from fadtk.fad import FrechetAudioDistance
    from fadtk.model_loader import get_all_models
    from fadtk.fad_batch import cache_embedding_files
    import fadtk.fad_batch as _fad_batch

    # Belt-and-suspenders: also replace Pool on fadtk's own module namespace.
    # multiprocessing.Pool is already _InlinePool (module-level patch), so any
    # `from multiprocessing import Pool` in fadtk.fad_batch already resolves to
    # _InlinePool. This line covers the edge case where fadtk was imported
    # before our patch ran (should not happen with lazy imports, but safe to keep).
    _fad_batch.Pool = _InlinePool

    models = {m.name: m for m in get_all_models()}
    if model_name not in models:
        available = sorted(models.keys())
        raise ValueError(
            f"Model '{model_name}' not found in fadtk. Available: {available}"
        )
    model = models[model_name]

    ref_path = Path(reference_dir)
    eval_path = Path(eval_dir)

    # Reference dir (preprocessed_unfiltered/) contains only audio files -- pass
    # directly so fadtk's .npy embedding cache is written adjacent to the audio
    # files on Drive and survives Colab session restarts.
    #
    # Eval dir has ACEStepPipeline _input_params.json sidecars alongside every
    # WAV. _make_wav_only_dir creates symlinks to audio files only in /tmp/
    # (local Colab fs supports symlinks even when targets are on Drive).
    ref_audio = ref_path
    eval_audio = _make_wav_only_dir(eval_path)

    print(f"[AffectScore] Caching embeddings for reference dir ({ref_audio.name})...")
    cache_embedding_files(ref_audio, model)

    print(f"[AffectScore] Caching embeddings for eval dir ({eval_audio.name})...")
    cache_embedding_files(eval_audio, model)

    print(f"[AffectScore] Computing FAD ({model_name})...")
    # audio_load_worker=0: load audio in the main thread (no DataLoader workers).
    # Avoids SemLock fork/spawn context mismatch when CUDA is active (Colab).
    fad = FrechetAudioDistance(model, audio_load_worker=0, load_model=False)
    score = fad.score(ref_audio, eval_audio)
    return float(score)


def run_alt_backbone_fad(
    audio_dirs: dict,
    reference_dir: str,
    backbone: str = None,
) -> dict:
    """Compute FAD with a non-MERT backbone across adapter variants.

    Addresses the MERT triple-role circularity: MERT generates corpus labels,
    measures FAD-MERT, and drives MER accuracy. An independent backbone validates
    that the two-tier FAD structure (no-lora vs LoRA variants) is not an artifact
    of MERT feature alignment.
    """
    if backbone is None:
        backbone = ALT_BACKBONE_MODELS[0]

    print(f"[AffectScore] === Alt Backbone FAD ({backbone}) ===")
    print(f"[AffectScore] Reference dir: {reference_dir}")
    print(f"[AffectScore] Variants: {list(audio_dirs.keys())}")

    results = {}
    for adapter_name, audio_dir in audio_dirs.items():
        print(f"[AffectScore] Computing FAD-{backbone} for adapter '{adapter_name}'...")
        try:
            fad_val = compute_fad_for_pair(reference_dir, audio_dir, backbone)
            results[adapter_name] = {"fad": fad_val, "backbone": backbone, "error": None}
            print(f"[AffectScore]   FAD-{backbone} ({adapter_name}): {fad_val:.4f}")
        except Exception as e:
            print(f"[AffectScore]   ERROR for '{adapter_name}': {e}")
            results[adapter_name] = {"fad": None, "backbone": backbone, "error": str(e)}

    backbone_slug = backbone.replace("/", "-").replace(" ", "_")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, f"alt_backbone_fad_{backbone_slug}.json")
    with open(results_path, "w") as f:
        json.dump({
            "backbone": backbone,
            "reference_dir": reference_dir,
            "results": results,
            "note": (
                "MERT-independent FAD backbone. "
                "Compare two-tier structure (no-lora vs LoRA) against FAD-MERT results. "
                "If the same tier structure holds under a different backbone, the finding "
                "is not an artifact of MERT representational circularity."
            ),
        }, f, indent=2)
    print(f"[AffectScore] Alt backbone FAD results saved to {results_path}")
    return results


def run_audio_metrics(
    audio_dir: str,
    reference_dir: str,
    adapter_name: str,
    held_out_path: str,
    reference_manifest_path: str,
    n_ref: int = 200,
    ref_seed: int = 42,
    alt_backbone: str = None,
) -> dict:
    """Run all audio quality metrics for one adapter variant.

    Computes FAD-MERT, CLAP-score, KLD (via PaSST), and PCE.
    Results are saved to eval/results/audio_metrics_{adapter_name}.json.
    """
    print(f"[AffectScore] === Audio Quality Metrics ===")
    print(f"[AffectScore] Adapter: {adapter_name}")
    print(f"[AffectScore] Audio dir: {audio_dir}")
    print(f"[AffectScore] Reference dir: {reference_dir}")

    print(f"[AffectScore] Loading held-out set from {held_out_path}")
    with open(held_out_path) as f:
        held_out = json.load(f)

    print(f"[AffectScore] Loading reference manifest from {reference_manifest_path}")
    with open(reference_manifest_path) as f:
        reference_clips = json.load(f)
    total_ref = len(reference_clips)
    if n_ref < total_ref:
        rng = np.random.default_rng(ref_seed)
        indices = sorted(rng.choice(total_ref, size=n_ref, replace=False).tolist())
        reference_clips = [reference_clips[i] for i in indices]
        print(f"[AffectScore] KLD reference: sampled {n_ref}/{total_ref} clips "
              f"(seed={ref_seed}). FAD uses full reference_dir.")
    else:
        print(f"[AffectScore] KLD reference: using all {total_ref} clips.")

    print("[AffectScore] Loading hear21passt (KLD)...")
    from hear21passt.base import load_model
    passt_model = load_model(mode="logits").cuda()
    passt_model.eval()

    print("[AffectScore] Loading msclap CLAP 2023 (CLAP-score)...")
    from msclap import CLAP
    clap_model = CLAP(version="2023", use_cuda=True)

    pce_values = []
    clap_values = []
    gen_probs_list = []
    skipped = 0
    evaluated = 0

    print("[AffectScore] Computing per-clip PCE, CLAP-score, and KLD probs...")
    for i, clip in enumerate(held_out):
        clip_id = clip["clip_id"]
        wav_path = os.path.join(audio_dir, f"{clip_id}.wav")

        if not os.path.exists(wav_path):
            skipped += 1
            continue

        valence = clip.get("V_A_valence", 0.0)
        arousal = clip.get("V_A_arousal", 0.0)
        style_prompt = _va_to_style_prompt(valence, arousal)

        try:
            pce_val = compute_pce(wav_path)
            pce_values.append(pce_val)

            clap_val = compute_clap_score(wav_path, style_prompt, clap_model)
            clap_values.append(clap_val)

            gen_probs = get_passt_probs(wav_path, passt_model)
            gen_probs_list.append(gen_probs)

            evaluated += 1
        except Exception as e:
            print(f"[AffectScore] ERROR on clip {clip_id}: {e}")
            skipped += 1
            continue

        if (i + 1) % 20 == 0:
            print(f"[AffectScore] Processed {i + 1}/{len(held_out)} clips "
                  f"({skipped} skipped)...")

    print(f"[AffectScore] Clips evaluated: {evaluated}, skipped: {skipped}")

    print("[AffectScore] Computing KLD reference distribution over training corpus...")
    ref_probs_list = []
    ref_skipped = 0
    for ref_clip in reference_clips:
        clip_id = ref_clip["clip_id"]
        ref_wav = os.path.join(reference_dir, f"{clip_id}.mp3")
        if not os.path.exists(ref_wav):
            ref_wav_alt = os.path.join(reference_dir, f"{clip_id}.wav")
            if os.path.exists(ref_wav_alt):
                ref_wav = ref_wav_alt
            else:
                ref_skipped += 1
                continue
        try:
            probs = get_passt_probs(ref_wav, passt_model)
            ref_probs_list.append(probs)
        except Exception as e:
            print(f"[AffectScore] ERROR on reference clip {clip_id}: {e}")
            ref_skipped += 1

    if not ref_probs_list:
        raise RuntimeError(
            f"No reference clips could be processed for KLD. "
            f"Check that reference_dir '{reference_dir}' contains WAV/MP3 files "
            f"matching clip_ids in {reference_manifest_path}"
        )

    print(f"[AffectScore] Reference probs computed: {len(ref_probs_list)} clips "
          f"({ref_skipped} skipped)")

    ref_probs = np.mean(np.stack(ref_probs_list, axis=0), axis=0)   # (527,)
    gen_probs = np.mean(np.stack(gen_probs_list, axis=0), axis=0)   # (527,)

    # KL(reference || generated) -- Stability-AI stable-audio-metrics convention
    kld = float(np.sum(ref_probs * np.log((ref_probs + EPS_KLD) / (gen_probs + EPS_KLD))))

    fad_vggish = None  # VGGish dropped: archived on GitHub, FAD-MERT is sole FAD metric

    print("[AffectScore] Computing FAD-MERT (primary metric)...")
    fad_mert = compute_fad_for_pair(reference_dir, audio_dir, MERT_MODEL_NAME)
    print(f"[AffectScore] FAD-MERT: {fad_mert:.4f}")

    fad_alt = None
    if alt_backbone:
        print(f"[AffectScore] Computing FAD with alt backbone '{alt_backbone}'...")
        try:
            fad_alt = compute_fad_for_pair(reference_dir, audio_dir, alt_backbone)
            print(f"[AffectScore] FAD-{alt_backbone}: {fad_alt:.4f}")
        except Exception as e:
            print(f"[AffectScore] Alt backbone FAD failed: {e}")

    result = {
        "adapter_name": adapter_name,
        "n_clips_evaluated": evaluated,
        "n_clips_skipped": skipped,
        "metrics": {
            "fad_mert": fad_mert,
            "fad_alt_backbone": fad_alt,
            "fad_alt_backbone_model": alt_backbone,
            "clap_score_mean": float(np.mean(clap_values)) if clap_values else None,
            "kld": kld,
            "pce_mean": float(np.mean(pce_values)) if pce_values else None,
        },
        "mert_is_primary": True,
        "kld_n_ref": len(reference_clips),
        "kld_ref_seed": ref_seed,
        "mert_model": MERT_MODEL_NAME,
        "notes": (
            "FAD-MERT is the sole FAD metric (Gui 2024; Tailleur 2024). "
            "fad_alt_backbone: MERT-circularity check (None if not requested)."
        ),
    }

    print(f"\n[AffectScore] === Audio Quality Metrics Summary ({adapter_name}) ===")
    print(f"  FAD-MERT (primary):  {fad_mert:.4f}")
    clap_str = (f"{result['metrics']['clap_score_mean']:.4f}"
                if result['metrics']['clap_score_mean'] is not None else "N/A")
    pce_str  = (f"{result['metrics']['pce_mean']:.4f}"
                if result['metrics']['pce_mean'] is not None else "N/A")
    print(f"  CLAP-score:          {clap_str}")
    print(f"  KLD:                 {kld:.4f}")
    print(f"  PCE (mean):          {pce_str}")
    print(f"  Clips: {evaluated} evaluated, {skipped} skipped")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"audio_metrics_{adapter_name}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[AffectScore] Results saved to {result_path}")

    return result


def _va_to_style_prompt(valence: float, arousal: float) -> str:
    """Derive a style prompt from V-A values for CLAP-score computation.

    held_out_set.json does not contain a style_prompt field.
    Verbatim copy of va_to_mood_words() from server/affectscore_server.py + suffix.
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
    mood_words = " ".join(parts)
    return mood_words + " orchestral game soundtrack"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute FAD-MERT, CLAP-score, KLD, and PCE "
            "for one adapter variant. Runs on Colab A100."
        )
    )
    parser.add_argument("--audio-dir", type=str, required=True)
    parser.add_argument("--reference-dir", type=str, required=True)
    parser.add_argument(
        "--adapter-name", type=str, required=True,
        choices=["full", "no-lora", "no-affect", "no-style", "r16", "r64"],
    )
    parser.add_argument(
        "--held-out", type=str,
        default=os.path.join(_REPO_ROOT, "data", "held_out_set.json"),
    )
    parser.add_argument(
        "--reference-manifest", type=str,
        default=os.path.join(_REPO_ROOT, "data", "training_set_clean_clap.json"),
    )
    parser.add_argument("--n-ref", type=int, default=200)
    parser.add_argument("--ref-seed", type=int, default=42)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument(
        "--backbone", type=str, default=None,
        help=(
            f"Run an additional FAD computation with this fadtk backbone model "
            f"to check MERT circularity. Suggested: '{ALT_BACKBONE_MODELS[0]}'."
        ),
    )
    args = parser.parse_args()

    tee = None
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        tee = _Tee(args.log_file)
        print(f"[AffectScore] Logging to {args.log_file}")

    try:
        run_audio_metrics(
            audio_dir=args.audio_dir,
            reference_dir=args.reference_dir,
            adapter_name=args.adapter_name,
            held_out_path=args.held_out,
            reference_manifest_path=args.reference_manifest,
            n_ref=args.n_ref,
            ref_seed=args.ref_seed,
            alt_backbone=args.backbone,
        )
    finally:
        if tee:
            tee.close()
