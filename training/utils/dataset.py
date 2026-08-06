import os
import json
import torch
import torchaudio


class AffectScoreDataset(torch.utils.data.Dataset):
    """PyTorch Dataset over a V-A annotated audio manifest.

    Each item returns a dict with keys:
      waveform  -- (1, N) float32 tensor, mono, at sample_rate
      valence   -- scalar float32 tensor in [-1, 1]
      arousal   -- scalar float32 tensor in [0, 1]
    """

    def __init__(self, manifest_path, audio_dir, sample_rate=44100, exclude_ids=None):
        with open(manifest_path, encoding="utf-8") as f:
            records = json.load(f)

        exclude_ids = exclude_ids or set()
        self.clips = [
            {
                "path":    os.path.join(audio_dir, r["filename"]),
                "valence": float(r["V_A_valence"]),
                "arousal": float(r["V_A_arousal"]),
            }
            for r in records
            if r["clip_id"] not in exclude_ids
        ]
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        waveform, sr = torchaudio.load(clip["path"])
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return {
            "waveform": waveform,
            "valence":  torch.tensor(clip["valence"], dtype=torch.float32),
            "arousal":  torch.tensor(clip["arousal"], dtype=torch.float32),
        }
