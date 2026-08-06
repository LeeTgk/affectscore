"""
Latency Benchmark -- 5x2 grid (step counts x chunk durations).
Extends gate1_rtf.py to a full benchmark grid: 50 trials per cell with 5
warm-up discarded, CUDA-synchronized client-side timing, peak VRAM capture,
and governor trigger rate computation.

Client-side timing: HTTP + generation + WAV encode + file write (not GPU-only).
Timer: torch.cuda.synchronize() -> t0 -> requests.post() -> file write -> t1 -> sync()

NOTE: torch.cuda.synchronize() here reflects server-side GPU completion only
if the benchmark and server share the same CUDA device (same Colab A100 instance
with server in a background thread). Server-side GPU completion is confirmed
by the HTTP response, which the server does not send until generation is complete.

Usage:
    # Start server first: python server/affectscore_server.py --lora <adapter_path>
    python eval/latency_bench.py
    python eval/latency_bench.py --n-trials 50 --n-warmup 5
    python eval/latency_bench.py --server-url http://127.0.0.1:8321
"""

import os
import sys
import json
import argparse
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

SERVER_URL = "http://127.0.0.1:8321"

STEP_COUNTS = [4, 8, 20, 40, 60]     # matches STEP_TIERS steps values
CHUNK_DURATIONS = [4.0, 8.0]          # seconds

N_TRIALS = 50                          # measured trials per cell
N_WARMUP = 5                           # warm-up trials discarded per cell

WAV_OUTPUT_PATH = os.path.join(
    _REPO_ROOT, "game", "audio", "_afs_buffer", "latency_bench_test.wav"
)
RESULTS_DIR = os.path.join(_HERE, "results")

# Must match STEP_TIERS in server/affectscore_server.py
STEP_TIERS = [
    {"steps": 4,  "label": "ultra-fast", "target_ms": 600},
    {"steps": 8,  "label": "real-time",  "target_ms": 1300},
    {"steps": 20, "label": "balanced",   "target_ms": 2800},
    {"steps": 40, "label": "eval",       "target_ms": 5500},
    {"steps": 60, "label": "quality",    "target_ms": 8000},
]
STEP_TIERS_BY_STEPS = {t["steps"]: t for t in STEP_TIERS}


def compute_p95(latencies_ms: list) -> float:
    """Return the 95th percentile of a list of latency measurements (ms)."""
    return float(np.percentile(latencies_ms, 95))


def _make_payload(steps: int, duration_s: float) -> dict:
    """Build a /generate request payload for benchmarking."""
    return {
        "affect_embedding": [0.0] * 512,
        "style_prompt": "neutral ambient orchestral",
        "chunk_duration_s": duration_s,
        "max_latency_ms": STEP_TIERS_BY_STEPS.get(steps, {}).get("target_ms", 10000),
    }


def _single_trial_latency_ms(payload: dict, server_url: str = SERVER_URL) -> float:
    """Measure one end-to-end trial latency in milliseconds.

    Client-side timing: HTTP + generation + WAV encode + file write.
    """
    import requests
    import torch

    os.makedirs(os.path.dirname(WAV_OUTPUT_PATH), exist_ok=True)

    # -- CLIENT-SIDE TIMING START (HTTP + generation + WAV encode + file write) --
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    resp = requests.post(f"{server_url}/generate", json=payload, timeout=120)
    resp.raise_for_status()
    wav_bytes = resp.content
    with open(WAV_OUTPUT_PATH, "wb") as f:
        f.write(wav_bytes)
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    # -- CLIENT-SIDE TIMING END --

    return (t1 - t0) * 1000.0


def run_grid_cell(steps: int, duration: float, n_trials: int = N_TRIALS,
                  n_warmup: int = N_WARMUP, server_url: str = SERVER_URL) -> dict:
    """Benchmark a single (steps, duration) grid cell.

    Runs n_warmup warm-up trials (discarded), then n_trials measured trials.

    Returns:
        dict with keys: steps, duration_s, mean_ms, std_ms, p95_ms, rtf,
                        peak_vram_mb, governor_trigger_rate, n_trials.
    """
    import torch

    payload = _make_payload(steps, duration)

    print(f"[AffectScore] Cell steps={steps}, duration={duration}s: "
          f"{n_warmup} warm-up + {n_trials} measured trials...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for i in range(n_warmup):
        try:
            lat = _single_trial_latency_ms(payload, server_url=server_url)
            print(f"[AffectScore]   Warm-up {i + 1}/{n_warmup}: {lat:.1f} ms")
        except Exception as e:
            print(f"[AffectScore]   Warm-up {i + 1} failed: {e}")

    latencies_ms = []
    for i in range(n_trials):
        try:
            lat = _single_trial_latency_ms(payload, server_url=server_url)
            latencies_ms.append(lat)
            if (i + 1) % 10 == 0:
                print(f"[AffectScore]   Trial {i + 1}/{n_trials}: {lat:.1f} ms")
        except Exception as e:
            print(f"[AffectScore]   Trial {i + 1} FAILED: {e}")

    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        torch.cuda.reset_peak_memory_stats()
    else:
        peak_vram_mb = 0.0

    if not latencies_ms:
        print(f"[AffectScore]   ERROR: No successful trials for cell steps={steps}, "
              f"duration={duration}s.")
        return {
            "steps": steps,
            "duration_s": duration,
            "mean_ms": float("nan"),
            "std_ms": float("nan"),
            "p95_ms": float("nan"),
            "rtf": float("nan"),
            "peak_vram_mb": peak_vram_mb,
            "governor_trigger_rate": float("nan"),
            "n_trials": 0,
        }

    mean_ms = float(np.mean(latencies_ms))
    std_ms = float(np.std(latencies_ms, ddof=1)) if len(latencies_ms) > 1 else 0.0
    p95_ms = compute_p95(latencies_ms)
    rtf = mean_ms / (duration * 1000.0)

    tier_target_ms = STEP_TIERS_BY_STEPS.get(steps, {}).get("target_ms", float("inf"))
    governor_trigger_rate = (
        sum(1 for lat in latencies_ms if lat > tier_target_ms) / len(latencies_ms)
    )

    print(f"[AffectScore]   steps={steps}, dur={duration}s: "
          f"mean={mean_ms:.1f}ms, p95={p95_ms:.1f}ms, RTF={rtf:.3f}, "
          f"VRAM={peak_vram_mb:.0f}MB, trigger={governor_trigger_rate:.2%}")

    return {
        "steps": steps,
        "duration_s": duration,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "p95_ms": p95_ms,
        "rtf": rtf,
        "peak_vram_mb": peak_vram_mb,
        "governor_trigger_rate": governor_trigger_rate,
        "n_trials": len(latencies_ms),
    }


def run_latency_grid(n_trials: int = N_TRIALS, n_warmup: int = N_WARMUP,
                     server_url: str = SERVER_URL) -> dict:
    """Run the full 5x2 benchmark grid (10 cells total).

    Each cell runs n_warmup warm-up trials (discarded) then n_trials measured
    trials with CUDA-synchronized client-side timing.

    Returns:
        dict with "grid_cells" list (one entry per cell) and metadata.
    """
    grid_cells = []

    total_cells = len(STEP_COUNTS) * len(CHUNK_DURATIONS)
    cell_idx = 0

    for steps in STEP_COUNTS:
        for duration in CHUNK_DURATIONS:
            cell_idx += 1
            print(f"\n[AffectScore] --- Cell {cell_idx}/{total_cells}: "
                  f"steps={steps}, duration={duration}s ---")
            cell_result = run_grid_cell(steps, duration, n_trials=n_trials,
                                        n_warmup=n_warmup, server_url=server_url)
            grid_cells.append(cell_result)

    return {
        "grid_cells": grid_cells,
        "step_counts": STEP_COUNTS,
        "chunk_durations": CHUNK_DURATIONS,
        "n_trials_per_cell": n_trials,
        "n_warmup_per_cell": n_warmup,
        "server_url": server_url,
        "timing_definition": (
            "Client-side: HTTP + generation + WAV encode + file write. "
            "CUDA-synchronized with torch.cuda.synchronize() before t0 and after t1."
        ),
    }


def _check_server(server_url: str = SERVER_URL):
    """Verify server is reachable and print status. Exits with code 1 on failure."""
    import requests

    try:
        health = requests.get(f"{server_url}/health", timeout=5).json()
        print(f"[AffectScore] Server status: {health.get('status')}, "
              f"device: {health.get('device')}")
    except Exception as e:
        print(f"[AffectScore] ERROR: Server not reachable at {server_url}: {e}")
        sys.exit(1)


def main():
    """Entry point: health check, run grid, save JSON, print summary table."""
    parser = argparse.ArgumentParser(
        description=(
            "Latency Benchmark: 5x2 grid of (step counts x chunk durations), "
            "50 trials per cell with 5 warm-up, CUDA-sync client-side timing."
        )
    )
    parser.add_argument("--n-trials", type=int, default=N_TRIALS,
                        help=f"Measured trials per grid cell (default: {N_TRIALS})")
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP,
                        help=f"Warm-up trials per grid cell, discarded (default: {N_WARMUP})")
    parser.add_argument("--server-url", type=str, default=SERVER_URL,
                        help=f"AffectScore server URL (default: {SERVER_URL})")
    args = parser.parse_args()

    active_server_url = args.server_url

    print(f"[AffectScore] Latency Benchmark")
    print(f"[AffectScore] Grid: {STEP_COUNTS} steps x {CHUNK_DURATIONS}s = "
          f"{len(STEP_COUNTS) * len(CHUNK_DURATIONS)} cells")
    print(f"[AffectScore] Trials: {args.n_warmup} warm-up + {args.n_trials} measured per cell")
    print(f"[AffectScore] Timing: client-side (HTTP + generation + WAV encode + file write)")
    print(f"[AffectScore] Server: {active_server_url}")

    _check_server(active_server_url)

    results = run_latency_grid(n_trials=args.n_trials, n_warmup=args.n_warmup,
                                server_url=active_server_url)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "latency_bench_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[AffectScore] Results saved to {results_path}")

    print(f"\n[AffectScore] === Latency Grid Summary ===")
    print(f"{'Steps':>6}  {'Dur(s)':>6}  {'Mean(ms)':>9}  {'Std(ms)':>8}  "
          f"{'p95(ms)':>8}  {'RTF':>6}  {'VRAM(MB)':>9}  {'TrigRate':>9}")
    print("-" * 77)
    for cell in results["grid_cells"]:
        print(f"{cell['steps']:>6}  {cell['duration_s']:>6.1f}  "
              f"{cell['mean_ms']:>9.1f}  {cell['std_ms']:>8.1f}  "
              f"{cell['p95_ms']:>8.1f}  {cell['rtf']:>6.3f}  "
              f"{cell['peak_vram_mb']:>9.1f}  {cell['governor_trigger_rate']:>9.2%}")


if __name__ == "__main__":
    main()
