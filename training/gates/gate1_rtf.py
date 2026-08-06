"""
Gate 1: Real-Time Factor (RTF) verification.
Measures client-side end-to-end latency for 4s audio chunk generation at 8 Turbo steps:
  HTTP POST start -> ACE-Step generation -> WAV encode -> WAV file write complete

Usage:
    python training/gates/gate1_rtf.py
    python training/gates/gate1_rtf.py --compile-test --n-trials 100
"""

import os
import sys
import json
import argparse
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

SERVER_URL = "http://127.0.0.1:8321"
CHUNK_DURATION_S = 4.0        # 4s fixed -- changing triggers recompile
TARGET_LATENCY_S = 2.0
WARMUP_INFERENCES = 2         # Run before stability count (compilation != recompile)
WAV_OUTPUT_PATH = os.path.join(
    _REPO_ROOT, "game", "audio", "_afs_buffer", "gate1_test.wav"
)


def _check_triton_windows():
    """Hard-fail on Windows if triton-windows is not installed.

    torch.compile requires triton. On Windows the upstream triton wheel does not
    work; triton-windows is the required alternative.
    """
    if sys.platform == "win32":
        try:
            import triton  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "triton-windows is required on Windows for torch.compile stability test.\n"
                "Install with: pip install triton-windows"
            )


def _make_payload(style_prompt: str = "neutral ambient game music") -> dict:
    """Construct a /generate request payload with a 512-d zero affect embedding."""
    return {
        "affect_embedding": [0.0] * 512,
        "style_prompt": style_prompt,
        "chunk_duration_s": CHUNK_DURATION_S,
        "max_latency_ms": 1800,  # governor selects 8 Turbo steps at 1100-1999ms
    }


def _single_request_latency(payload: dict) -> float:
    """Measure one end-to-end request and write WAV to VFS path.

    Timer starts BEFORE requests.post() (client-side timing).
    Timer ends AFTER open(WAV_OUTPUT_PATH, 'wb').write(wav_bytes).
    This includes HTTP roundtrip + generation + WAV encode + file write.
    """
    import requests

    os.makedirs(os.path.dirname(WAV_OUTPUT_PATH), exist_ok=True)

    t0 = time.perf_counter()
    resp = requests.post(f"{SERVER_URL}/generate", json=payload, timeout=30)
    resp.raise_for_status()
    wav_bytes = resp.content
    with open(WAV_OUTPUT_PATH, "wb") as f:
        f.write(wav_bytes)
    t1 = time.perf_counter()

    return t1 - t0


def run_latency_test(n_trials: int = 20, style_prompt: str = "neutral ambient game music"):
    """Run RTF latency measurement over n_trials requests.

    Returns:
        dict with mean_latency_s, rtf, pass, all_latencies_s, target_s
    """
    import requests

    print(f"[AffectScore] Gate 1 RTF test: {n_trials} trials at {CHUNK_DURATION_S}s chunk")
    print(f"[AffectScore] Server: {SERVER_URL}")
    print(f"[AffectScore] WAV output: {WAV_OUTPUT_PATH}")

    try:
        health = requests.get(f"{SERVER_URL}/health", timeout=5).json()
        print(f"[AffectScore] Server status: {health.get('status')}, "
              f"device: {health.get('device')}")
    except Exception as e:
        print(f"[AffectScore] ERROR: Server not reachable at {SERVER_URL}: {e}")
        sys.exit(1)

    payload = _make_payload(style_prompt)
    latencies = []

    for i in range(n_trials):
        try:
            latency_s = _single_request_latency(payload)
        except Exception as e:
            print(f"[AffectScore] Trial {i + 1} FAILED: {e}")
            continue

        rtf = latency_s / CHUNK_DURATION_S
        latencies.append(latency_s)
        print(f"[AffectScore] Trial {i + 1:3d}/{n_trials}: "
              f"latency={latency_s:.3f}s  RTF={rtf:.3f}")

    if not latencies:
        print("[AffectScore] Gate 1 FAILED: No successful trials.")
        sys.exit(1)

    mean_latency = sum(latencies) / len(latencies)
    gate_pass = mean_latency < TARGET_LATENCY_S

    print(f"\n[AffectScore] === Gate 1 RTF Results ===")
    print(f"  Mean latency: {mean_latency:.3f} s (target: < {TARGET_LATENCY_S} s)")
    print(f"  RTF:          {mean_latency / CHUNK_DURATION_S:.3f}")
    print(f"  Result:       {'PASS' if gate_pass else 'FAIL'}")

    return {
        "pass": gate_pass,
        "mean_latency_s": mean_latency,
        "rtf": mean_latency / CHUNK_DURATION_S,
        "target_s": TARGET_LATENCY_S,
        "chunk_duration_s": CHUNK_DURATION_S,
        "n_trials": len(latencies),
        "all_latencies_s": latencies,
    }


def run_compile_stability_test(n_inferences: int = 100):
    """Test torch.compile stability over n_inferences at fixed chunk duration.

    Procedure:
    1. Run WARMUP_INFERENCES warmup calls (compilation != recompile)
    2. Run n_inferences stability calls at fixed CHUNK_DURATION_S
    3. Count recompile events in TORCH_LOGS output
    4. Pass condition: recompile_count == 0 (< 1 per 100 inferences)

    The server must have been started with torch.compile enabled and
    TORCH_LOGS=recompiles set in its environment before startup.

    Returns:
        dict with recompile_count, pass, n_inferences
    """
    _check_triton_windows()
    import requests

    print(f"[AffectScore] Gate 1 compile stability test: {n_inferences} inferences "
          f"at fixed {CHUNK_DURATION_S}s")
    print(f"[AffectScore] Running {WARMUP_INFERENCES} warmup inferences first...")

    payload = _make_payload("neutral ambient game music")

    for i in range(WARMUP_INFERENCES):
        try:
            latency_s = _single_request_latency(payload)
            print(f"[AffectScore] Warmup {i + 1}/{WARMUP_INFERENCES}: {latency_s:.3f}s")
        except Exception as e:
            print(f"[AffectScore] Warmup {i + 1} failed: {e}")

    print(f"[AffectScore] Warmup complete. Starting stability measurement...")

    for i in range(n_inferences):
        try:
            latency_s = _single_request_latency(payload)
            if (i + 1) % 20 == 0:
                print(f"[AffectScore] Stability inference {i + 1}/{n_inferences}: "
                      f"{latency_s:.3f}s")
        except Exception as e:
            print(f"[AffectScore] Stability inference {i + 1} failed: {e}")

    # recompile_count must be measured from the server log, not hardcoded to 0.
    # Set TORCH_LOGS=recompiles in the server process environment BEFORE starting the server.
    # Recompile events appear as lines containing "Recompiling" in server stdout.
    # We mark this MANUAL_CHECK_REQUIRED so GATE-REPORT.md fill-in forces the real number.
    recompile_count = "MANUAL_CHECK_REQUIRED"
    print(f"\n[AffectScore] Stability test complete.")
    print(f"[AffectScore] MANUAL ACTION REQUIRED: Count recompile events from server log.")
    print(f"[AffectScore] Steps:")
    print(f"[AffectScore]   1. Start server with TORCH_LOGS=recompiles env var set")
    print(f"[AffectScore]   2. Redirect server output: server.py ... > server_log.txt 2>&1")
    print(f"[AffectScore]   3. After this test: grep -c Recompiling server_log.txt")
    print(f"[AffectScore]   4. Record count in training/gates/GATE-REPORT.md")
    print(f"[AffectScore] Pass condition: recompile_count == 0 (< 1 per 100 inferences)")

    # gate_pass is False until manual verification confirms 0 recompiles.
    gate_pass = False

    return {
        "pass": gate_pass,
        "recompile_count": recompile_count,
        "n_inferences": n_inferences,
        "warmup_inferences": WARMUP_INFERENCES,
        "chunk_duration_s": CHUNK_DURATION_S,
        "note": (
            "MANUAL_CHECK_REQUIRED: start server with TORCH_LOGS=recompiles env var, "
            "redirect stdout to server_log.txt, then run: grep -c Recompiling server_log.txt. "
            "Record the count in training/gates/GATE-REPORT.md."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gate 1: RTF latency measurement and torch.compile stability test"
    )
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Number of RTF measurement trials (default: 20)")
    parser.add_argument("--compile-test", action="store_true",
                        help="Run torch.compile stability test (100 inferences at fixed 4s)")
    parser.add_argument("--compile-n", type=int, default=100,
                        help="Number of inferences for compile stability test (default: 100)")
    parser.add_argument("--style-prompt", type=str,
                        default="neutral ambient game music",
                        help="Style prompt for generation requests")
    args = parser.parse_args()

    results = {}

    rtf_results = run_latency_test(args.n_trials, args.style_prompt)
    results["rtf"] = rtf_results

    if args.compile_test:
        compile_results = run_compile_stability_test(args.compile_n)
        results["compile_stability"] = compile_results

    results["gate1_pass"] = rtf_results["pass"]
    if args.compile_test:
        results["gate1_pass"] = rtf_results["pass"] and compile_results["pass"]

    results_path = os.path.join(_HERE, "gate1_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[AffectScore] Gate 1 results saved to {results_path}")
    print(f"[AffectScore] Gate 1: {'PASS' if results['gate1_pass'] else 'FAIL'}")

    if not results["gate1_pass"]:
        sys.exit(1)
