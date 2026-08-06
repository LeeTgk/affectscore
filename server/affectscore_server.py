"""
FastAPI server wrapping the LoRA-tuned ACE-Step pipeline for real-time music chunk generation.
Start with: python server/affectscore_server.py --port 8321 --device-id 0
"""

import argparse
import os
import time
import tempfile
import uuid
from typing import List, Optional

import torch
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

# Bypass torchaudio's torchcodec path — requires FFmpeg DLLs not present on Windows.
# Newer torchaudio routes torchaudio.save() through torchcodec; patch to use soundfile directly.
try:
    import torchaudio as _torchaudio
    import soundfile as _sf

    import numpy as _np
    import torch as _torch

    def _soundfile_save(filepath, src, sample_rate, *args, **kwargs):
        data = src.numpy()
        if data.ndim == 2:
            data = data.T  # (C, N) -> (N, C) for soundfile
        # Normalize to avoid clipping; PCM_16 required by Python's wave module.
        peak = _np.abs(data).max()
        if peak > 1e-6:
            data = data / max(peak, 1.0)
        _sf.write(str(filepath), data, int(sample_rate), subtype='PCM_16')

    def _soundfile_load(filepath, *args, **kwargs):
        data, sr = _sf.read(str(filepath), always_2d=True)  # (N, C)
        waveform = _torch.from_numpy(data.T.astype(_np.float32))  # (C, N)
        return waveform, sr

    _torchaudio.save = _soundfile_save
    _torchaudio.load = _soundfile_load
    print("[DiT] torchaudio.save/load patched to use soundfile (torchcodec bypass)")
except Exception as _e:
    print(f"[DiT] WARNING: torchaudio patch failed: {_e}")

try:
    from acestep.pipeline_ace_step import ACEStepPipeline
    _ACE_STEP_AVAILABLE = True
except ImportError:
    _ACE_STEP_AVAILABLE = False
    ACEStepPipeline = None


def _check_flash_attention():
    """Hard-fail if flash_attn is not installed."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "[DiT] flash_attn not installed. "
            "Install with: pip install flash-attn --no-build-isolation"
        )


def va_to_mood_words(valence: float, arousal: float) -> str:
    """Map V-A values to CLAP-compatible text string.

    Copied verbatim from training/train_lora.py to keep inference conditioning
    consistent with training conditioning.
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

    Canonical reference copy — thresholds must match eval/generate_eval_set.py verbatim.
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


SAMPLE_RATE = 48000  # ACE-Step outputs 48 kHz
CHANNELS = 2

# Latency targets per diffusion step count.
# 8 steps = real-time; 40 steps = evaluation default; 60 steps = quality ceiling.
STEP_TIERS = [
    {"steps": 4,  "label": "ultra-fast", "target_ms": 600},
    {"steps": 8,  "label": "real-time",  "target_ms": 1300},
    {"steps": 20, "label": "balanced",   "target_ms": 2800},
    {"steps": 40, "label": "eval",       "target_ms": 5500},
    {"steps": 60, "label": "quality",    "target_ms": 8000},
]


class GenerateRequest(BaseModel):
    affect_embedding: List[float] = Field(..., max_length=512)  # DoS guard
    style_prompt: str
    chunk_duration_s: float = 4.0
    max_latency_ms: float = 1800
    valence: float = 0.0
    arousal: float = 0.5
    ref_audio_path: Optional[str] = None
    ref_audio_strength: float = 0.5
    seed: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    gpu_util: float
    last_step_count: int
    device: str
    lora_path: str


class AffectScoreDiT:
    """Wraps ACEStepPipeline for real-time chunk generation."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.pipeline = None
        self.last_step_count = 8
        self.sample_rate = SAMPLE_RATE
        self.lora_path = "none"
        self.lora_weight = 0.3

    def load(self, checkpoint_dir: str, lora_path: str, torch_compile: bool = False,
             skip_flash_check: bool = False):
        """Load ACEStepPipeline and optionally set LoRA adapter path."""
        if not _ACE_STEP_AVAILABLE:
            raise RuntimeError(
                "[DiT] ACE-Step not installed. Run: pip install -e ace_step/"
            )

        if skip_flash_check:
            print("[DiT] WARNING: flash-attn check skipped (--no-flash-check). "
                  "Latency measurement only — do NOT use for production training.")
        else:
            _check_flash_attention()

        print(f"[DiT] Loading ACEStepPipeline on device cuda:{self.device_id}")
        print(f"[DiT] Checkpoint dir: {checkpoint_dir or '(HF download)'}")

        self.pipeline = ACEStepPipeline(
            checkpoint_dir=checkpoint_dir if checkpoint_dir else None,
            device_id=self.device_id,
            dtype="bfloat16",
            torch_compile=torch_compile,
            cpu_offload=False,   # keep in VRAM between requests
        )
        self.pipeline.load_checkpoint()

        if lora_path and lora_path != "none" and os.path.exists(lora_path):
            lora_abs = os.path.abspath(lora_path)
            if os.path.isdir(lora_abs):
                expected = os.path.join(lora_abs, "pytorch_lora_weights.safetensors")
                if not os.path.isfile(expected):
                    raise ValueError(
                        f"[DiT] LoRA directory missing pytorch_lora_weights.safetensors: {lora_abs}"
                    )
            elif not lora_abs.endswith((".safetensors", ".pt")):
                raise ValueError(
                    f"[DiT] Invalid LoRA path: {lora_path}. Must be a .safetensors/.pt file or a directory."
                )
            self.lora_path = lora_abs
            print(f"[DiT] LoRA adapter: {self.lora_path}")
        else:
            self.lora_path = "none"
            print("[DiT] No LoRA adapter (base model inference)")

        print("[DiT] ACEStepPipeline ready.")

    def select_step_count(self, max_latency_ms: float) -> int:
        """Latency governor: pick highest-quality step tier within budget."""
        for tier in reversed(STEP_TIERS):
            if tier["target_ms"] <= max_latency_ms:
                self.last_step_count = tier["steps"]
                return tier["steps"]
        self.last_step_count = STEP_TIERS[0]["steps"]
        return STEP_TIERS[0]["steps"]

    @torch.inference_mode()
    def generate(
        self,
        affect_embedding: List[float],
        style_prompt: str,
        chunk_duration_s: float,
        num_steps: int,
        ref_audio_path: Optional[str] = None,
        ref_audio_strength: float = 0.5,
        valence: float = 0.0,
        arousal: float = 0.5,
        seed: Optional[int] = None,
    ) -> bytes:
        """Generate a music chunk and return WAV bytes."""
        output_path = os.path.join(
            tempfile.mkdtemp(), f"afs_{uuid.uuid4().hex[:8]}.wav"
        )

        mood_text = va_to_mood_words(valence, arousal)
        full_prompt = f"{mood_text}, {style_prompt}"

        embed_norm = (
            sum(v * v for v in affect_embedding) ** 0.5
            if affect_embedding else 0.0
        )
        embed_head = [round(v, 4) for v in affect_embedding[:4]]

        use_continuation = (
            ref_audio_path is not None
            and ref_audio_path.endswith(".wav")
            and os.path.isfile(ref_audio_path)
        )
        print(
            f"[DiT] Generating {chunk_duration_s}s, {num_steps} steps, "
            f"prompt='{full_prompt[:60]}', "
            f"embed_norm={embed_norm:.4f} head={embed_head}"
            + (" [continuation]" if use_continuation else "")
        )

        self.pipeline(
            audio_duration=chunk_duration_s,
            prompt=full_prompt,
            lyrics="[instrumental]",
            infer_step=num_steps,
            guidance_scale=7.0,
            scheduler_type="euler",
            cfg_type="apg",
            omega_scale=10.0,
            manual_seeds=[seed] if seed is not None else None,
            guidance_interval=0.5,
            guidance_interval_decay=0.0,
            min_guidance_scale=3.0,
            use_erg_tag=True,
            use_erg_lyric=True,
            use_erg_diffusion=True,
            lora_name_or_path=self.lora_path,
            lora_weight=self.lora_weight,
            save_path=output_path,
            audio2audio_enable=use_continuation,
            ref_audio_input=ref_audio_path if use_continuation else None,
            ref_audio_strength=ref_audio_strength if use_continuation else 0.5,
        )

        with open(output_path, "rb") as f:
            wav_bytes = f.read()
        os.unlink(output_path)
        return wav_bytes

    def get_gpu_utilization(self) -> float:
        """Return current GPU memory utilization as a fraction."""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.device_id}")
            mem = torch.cuda.memory_allocated(device)
            total = torch.cuda.get_device_properties(device).total_memory
            return mem / total if total > 0 else 0.0
        return 0.0


app = FastAPI(title="AffectScore DiT Server", version="0.2.0")
dit_model: Optional[AffectScoreDiT] = None


@app.on_event("startup")
async def startup():
    global dit_model
    dit_model = AffectScoreDiT(device_id=app.state.device_id)
    dit_model.lora_weight = getattr(app.state, "lora_weight", 0.3)
    dit_model.load(
        checkpoint_dir=app.state.checkpoint_dir,
        lora_path=app.state.lora_path,
        torch_compile=app.state.torch_compile,
        skip_flash_check=app.state.no_flash_check,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    if dit_model is None:
        return HealthResponse(status="loading", gpu_util=0.0,
                               last_step_count=0, device="unknown", lora_path="none")
    return HealthResponse(
        status="ok",
        gpu_util=dit_model.get_gpu_utilization(),
        last_step_count=dit_model.last_step_count,
        device=f"cuda:{dit_model.device_id}",
        lora_path=dit_model.lora_path,
    )


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate a music chunk and return raw WAV bytes."""
    t0 = time.time()

    num_steps = dit_model.select_step_count(req.max_latency_ms)
    wav_bytes = dit_model.generate(
        affect_embedding=req.affect_embedding,
        style_prompt=req.style_prompt,
        chunk_duration_s=req.chunk_duration_s,
        num_steps=num_steps,
        valence=req.valence,
        arousal=req.arousal,
        ref_audio_path=req.ref_audio_path,
        ref_audio_strength=req.ref_audio_strength,
        seed=req.seed,
    )

    gen_ms = (time.time() - t0) * 1000
    print(f"[DiT] Chunk generated in {gen_ms:.0f}ms "
          f"({num_steps} steps, {req.chunk_duration_s}s)")

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Generation-Time-Ms": str(int(gen_ms))},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AffectScore DiT Server")
    parser.add_argument("--port", type=int, default=8321)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="",
                        help="Local ACE-Step checkpoint dir (empty = HF download)")
    parser.add_argument("--lora", type=str, default="none",
                        help="Path to LoRA adapter .safetensors or .pt (default: none)")
    parser.add_argument("--torch-compile", action="store_true",
                        help="Enable torch.compile on the transformer")
    parser.add_argument("--chunk-duration", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=8,
                        help="Default Turbo inference steps")
    parser.add_argument("--no-flash-check", action="store_true",
                        help="Skip flash-attn startup check (latency testing only)")
    parser.add_argument("--lora-weight", type=float, default=0.3,
                        help="LoRA adapter blending weight (0=off, 1=full; default 0.3)")
    args = parser.parse_args()

    app.state.device_id = args.device_id
    app.state.checkpoint_dir = args.checkpoint_dir
    app.state.lora_path = args.lora
    app.state.torch_compile = args.torch_compile
    app.state.chunk_duration = args.chunk_duration
    app.state.default_steps = args.steps
    app.state.no_flash_check = args.no_flash_check
    app.state.lora_weight = args.lora_weight

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")
