# AffectScore

Real-time adaptive music generation for narrative-driven games. AffectScore uses a LoRA-adapted ACE-Step latent diffusion transformer conditioned on a two-layer signal: authored designer intent (Layer 1) and runtime player engagement telemetry (Layer 2). Evaluated inside a running Ren'Py visual novel engine.

**Paper:** AffectScore: Real-Time Adaptive Music Generation for Narrative Games via Two-Layer Affect Conditioning — under review.

## Repository layout

| Directory | Contents |
|-----------|----------|
| `server/` | FastAPI generation server (ACE-Step inference, port 8321) |
| `game/affectscore/` | Ren'Py-side Python package: encoder, orchestrator, signal collector |
| `training/` | LoRA fine-tuning scripts, gate diagnostics, Colab setup |
| `eval/` | Five computational evaluation components |
| `colab/` | Jupyter notebooks for running gate checks, training, and evaluation on Colab |

## Model weights and dataset

- **LoRA checkpoints** (6 variants):
  - [affectscore-ace-step-r32-20260629](https://huggingface.co/HiiragiLee/affectscore-ace-step-r32-20260629) — full model, r=32 (selected)
  - [affectscore-ace-step-r16-20260629](https://huggingface.co/HiiragiLee/affectscore-ace-step-r16-20260629) — rank sweep, r=16
  - [affectscore-ace-step-r64-20260629](https://huggingface.co/HiiragiLee/affectscore-ace-step-r64-20260629) — rank sweep, r=64 (diverged)
  - [affectscore-ace-step-r32-20260629-no-style](https://huggingface.co/HiiragiLee/affectscore-ace-step-r32-20260629-no-style) — ablation: no style text descriptor + CLAP auxiliary loss disabled
  - [affectscore-ace-step-r32-20260629-no-affect](https://huggingface.co/HiiragiLee/affectscore-ace-step-r32-20260629-no-affect) — ablation: no Layer 1 V-A conditioning
  - [affectscore-ace-step-r32-20260629-no-lora](https://huggingface.co/HiiragiLee/affectscore-ace-step-r32-20260629-no-lora) — vanilla ACE-Step baseline
- **Training dataset** (2,938 clips, CC0/CC-BY): [https://zenodo.org/records/21830658](https://zenodo.org/records/21830658)

## Quick start

```bash
# Server (GPU required)
pip install -r requirements.txt
python server/affectscore_server.py --lora <adapter_dir>

# Latency benchmark (server must be running)
python eval/latency_bench.py

# Evaluation (requires trained checkpoint and Zenodo dataset in data/)
python eval/generate_eval_set.py --lora <adapter_dir> --audio-dir data/preprocessed --out eval/results/eval_set
python eval/audio_metrics.py
python eval/emotion_classify.py
python eval/temporal_coherence.py
python eval/simulated_archetypes.py
```

## Colab notebooks

Three notebooks in `colab/` reproduce the full pipeline on a free Colab A100:

| Notebook | Purpose |
|----------|---------|
| `AffectScore - Gates.ipynb` | Run gate checks (RTF, attention entropy, lambda calibration) before training |
| `AffectScore - Training.ipynb` | Dataset curation, rank-sweep training, ablation variants, SAO comparison |
| `AffectScore - Eval.ipynb` | All five evaluation components: latency, audio quality, MER, temporal coherence, simulated archetypes |

Open the notebook, add your Hugging Face token as a Colab secret named `HF_TOKEN`, and run cells top to bottom. Cell 1 clones the repository from GitHub automatically.

Notebooks were developed and validated on the Colab 2025.10 runtime (Python 3.11, CUDA 12.4, PyTorch 2.5). Other runtime versions may require dependency adjustments.

## Two-layer conditioning

Layer 1 (designer intent, authored at scene-creation time): `scene_valence`, `scene_arousal`, `arc_position` on the Russell circumplex.

Layer 2 (player engagement, captured at runtime from behavioral trace): `choice_latency_norm`, `dwell_deviation_norm`, `interaction_rate_norm` — modulates music intensity and texture around the Layer 1 anchor without overriding it.

## Citation

Citation will be added once the paper is published. If you use this code before then, please link to this repository.

## License

Code: MIT. Training dataset: CC0. LoRA weights are derived from ACE-Step — see [ACE-Step license](https://github.com/ace-step/ACE-Step).
