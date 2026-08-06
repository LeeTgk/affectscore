# game/affectscore/inference_client.py
import json
import time


class _InferenceClientImpl:
    """
    Communicates with the local DiT server via HTTP using only stdlib (no requests).

    POST /generate expects {affect_embedding, style_prompt, chunk_duration_s, max_latency_ms}.
    GET /health returns {status, gpu_util, last_step_count}.
    All calls are intended to run from a background daemon thread.
    """

    def __init__(self, base_url):
        self._base_url = base_url
        self._healthy = False

    def check_health(self):
        """Ping the server. Returns dict or None."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self._base_url}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                data = json.loads(resp.read().decode())
                self._healthy = data.get("status") == "ok"
                return data
        except Exception:
            self._healthy = False
            return None

    def generate_chunk(self, affect_embedding, style_prompt,
                       chunk_duration_s, max_latency_ms,
                       valence=0.0, arousal=0.5,
                       ref_audio_path=None, ref_audio_strength=0.5,
                       seed=None):
        """Request a music chunk from the DiT server.

        Returns:
            Tuple of (wav_bytes, generation_time_ms) on success,
            or (None, None) on failure.
        """
        try:
            import urllib.request

            payload = json.dumps({
                "affect_embedding": affect_embedding,
                "style_prompt": style_prompt,
                "chunk_duration_s": chunk_duration_s,
                "max_latency_ms": max_latency_ms,
                "valence": valence,
                "arousal": arousal,
                "ref_audio_path": ref_audio_path,
                "ref_audio_strength": ref_audio_strength,
                "seed": seed,
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self._base_url}/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            # Timeout scales with clip length: the STEP_TIERS target_ms was
            # measured on 4-second clips; longer clips take proportionally longer.
            # Formula: target_ms × (clip_dur / 4 s) + 10 s safety margin.
            scaled_timeout = (max_latency_ms / 1000.0) * max(1.0, chunk_duration_s / 4.0) + 10.0
            t0 = time.time()
            with urllib.request.urlopen(
                req, timeout=scaled_timeout
            ) as resp:
                wav_bytes = resp.read()
                gen_time_ms = (time.time() - t0) * 1000.0
                return wav_bytes, gen_time_ms

        except Exception as e:
            print(f"[AffectScore] Generation request failed: {e}")
            return None, None
