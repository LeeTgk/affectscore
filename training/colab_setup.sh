#!/bin/bash
# training/colab_setup.sh -- Google Colab Pro+ A100 session setup for AffectScore training.
# Usage: bash training/colab_setup.sh [--eval]
# Prerequisites: mount Google Drive and set HF_TOKEN in a notebook cell before running.
set -e

INSTALL_EVAL=0
for arg in "$@"; do
    case "$arg" in
        --eval) INSTALL_EVAL=1 ;;
    esac
done

DRIVE="/content/drive/MyDrive/affectscore"
REPO_DIR="/content/affectscore"

echo "[AffectScore] === Colab Session Setup ==="

if [ -f "$REPO_DIR/training/colab_setup.sh" ]; then
    if git -C "$REPO_DIR" rev-parse --git-dir > /dev/null 2>&1; then
        echo "[AffectScore] Git repo detected — pulling latest..."
        git -C "$REPO_DIR" pull --ff-only
    else
        echo "[AffectScore] Zip extraction detected (no .git) — skipping clone."
    fi
else
    echo "[AffectScore] Cloning repo..."
    git clone https://github.com/LeeTgk/affectscore.git "$REPO_DIR"
fi
cd "$REPO_DIR"

if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "[AffectScore] Initialising ACE-Step submodule..."
    git submodule update --init --recursive
else
    echo "[AffectScore] Cloning ACE-Step directly (zip workflow — no .git)..."
    if [ ! -f "ace_step/setup.py" ]; then
        git clone https://github.com/ace-step/ACE-Step ace_step/
    else
        echo "[AffectScore] ace_step/ already present — skipping."
    fi
fi

echo "[AffectScore] Symlinking HF cache to Drive..."
mkdir -p "$DRIVE/models/hf-cache"
ln -sfn "$DRIVE/models/hf-cache" "$HOME/.cache/huggingface"

echo "[AffectScore] Installing Python dependencies..."
pip install -e ace_step/ -q
pip install peft -q
# msclap declares numpy<2.0.0 but its runtime code is fully compatible with numpy 2.x.
# Installing with --no-deps prevents pip from downgrading numpy and breaking Colab's
# pre-compiled scipy/sklearn/diffusers (all require numpy 2.x binary ABI).
pip install msclap --no-deps -q
# msclap calls importlib.metadata.version("laion_clap") at import time -- the package
# must be installed with its metadata present. laion_clap also declares numpy<2
# so install with --no-deps; all its real deps are already present in Colab.
pip install laion_clap --no-deps -q
pip install torchlibrosa -q
# Pre-built wheel available for torch 2.8+cu126 (tested: flash_attn 2.8.3.post1).
# --find-links supplements PyPI; pip picks the matching wheel automatically.
# Fallback: if no wheel found, MAX_JOBS=4 compiles in ~15 min.
pip install flash-attn \
    --find-links https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.7.4 \
    --no-build-isolation -q || \
    (MAX_JOBS=4 pip install flash-attn --no-build-isolation)
pip install triton-windows -q 2>/dev/null || true
# Editable install path hook unreliable in Colab kernels -- write a .pth file instead.
python -c "
import site, os
pth = os.path.join(site.getsitepackages()[0], 'acestep.pth')
with open(pth, 'w') as f:
    f.write('/content/affectscore/ace_step\n')
print('[AffectScore] Wrote ' + pth)
"

echo "[AffectScore] Verifying installation..."
python -c "from acestep.pipeline_ace_step import ACEStepPipeline; print('[AffectScore] ACE-Step import: OK')"
python -c "import flash_attn; print('[AffectScore] Flash Attention import: OK')"
python -c "import torch; supported = torch.cuda.is_bf16_supported(); print(f'[AffectScore] BF16 supported: {supported}'); assert supported, 'BF16 not supported — A100 required'"

echo "[AffectScore] === Setup complete. Ready for training. ==="

if [ "$INSTALL_EVAL" = "1" ]; then
    echo "[AffectScore] --eval: Installing evaluation dependencies..."

    # fadtk 1.1.0 declares numpy<2 but is source-compatible with 2.x;
    # installing with deps would downgrade numpy and break ace-step + flash-attn.
    pip install fadtk --no-deps -q
    pip install soundfile hypy_utils -q

    # hear21passt: same numpy<2 constraint; add timm + einops for PaSST attention.
    pip install hear21passt --no-deps -q
    pip install timm einops -q

    # Restore librosa==0.11.0 -- pip may have downgraded it while resolving
    # fadtk/hear21passt deps. ace-step requires exactly 0.11.0.
    pip install "librosa==0.11.0" -q
    python -c "import librosa; print(f'[AffectScore] librosa {librosa.__version__}: OK')"

    python -c "from scipy.stats import wilcoxon; print('[AffectScore] scipy.stats: OK')"

    # GIT_TERMINAL_PROMPT=0: prevents git from opening a credential prompt on the
    # headless Colab terminal (public repo, no auth needed -- prompt just hangs/fails).
    if [ -d "$REPO_DIR/music2emo_repo" ]; then
        echo "[AffectScore] music2emo_repo/ already present."
    elif [ -d "$DRIVE/music2emo_repo" ]; then
        echo "[AffectScore] Copying music2emo_repo from Drive..."
        cp -r "$DRIVE/music2emo_repo" "$REPO_DIR/music2emo_repo"
    else
        echo "[AffectScore] Cloning music2emo..."
        # -c credential.helper="" overrides any Colab-configured credential helper
        # that tries to prompt on a headless terminal and hangs/fails for public repos.
        GIT_TERMINAL_PROMPT=0 git -c credential.helper="" clone --depth=1 \
            https://github.com/AMAAI-Lab/Music2Emotion.git "$REPO_DIR/music2emo_repo"
    fi
    pip install -e "$REPO_DIR/music2emo_repo/" -q 2>/dev/null || true
    pip install mir_eval pretty_midi jams hydra-core omegaconf -q

    # Pre-download MERT-v1-95M into the Drive-backed HF cache.
    # fadtk spawns multiprocessing workers to cache embeddings; if the model
    # isn't already in ~/.cache/huggingface the workers race to download it and
    # crash with "does not appear to have a file named pytorch_model.bin" because
    # m-a-p/MERT-v1-95M needs trust_remote_code=True to resolve its custom
    # architecture -- which fadtk's subprocess context may not pass.
    echo "[AffectScore] Pre-downloading MERT-v1-95M (trust_remote_code=True)..."
    python -c "
from transformers import AutoModel, Wav2Vec2FeatureExtractor
try:
    Wav2Vec2FeatureExtractor.from_pretrained('m-a-p/MERT-v1-95M', trust_remote_code=True)
    AutoModel.from_pretrained('m-a-p/MERT-v1-95M', trust_remote_code=True)
    print('[AffectScore] MERT-v1-95M pre-download: OK')
except Exception as e:
    print(f'[AffectScore] WARNING: MERT pre-download failed: {e}')
    print('[AffectScore] FAD-MERT will attempt download at eval time.')
"

    python -c "
import fadtk; print('[AffectScore] fadtk: OK')
import librosa; print('[AffectScore] librosa: OK')
from scipy.stats import wilcoxon; print('[AffectScore] scipy.stats: OK')
"
    echo "[AffectScore] === Eval dependencies ready. ==="
fi
