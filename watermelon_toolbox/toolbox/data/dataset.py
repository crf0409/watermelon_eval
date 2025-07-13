from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

import librosa
import numpy as np
import torch
from PIL import Image
import yaml

from .transforms import image_augment, RandomCropAudio


class WatermelonDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root: Path,
                 split: str = "train",
                 audio_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
                 sr: int = 16000):
        self.root = Path(root)
        self.split = split
        self.sr = sr
        with open(self.root / "splits.yaml", "r") as f:
            splits = yaml.safe_load(f)
        self.folds: List[str] = splits[split]
        self.audio_transform = audio_transform or RandomCropAudio()
        self.image_transform = image_transform or image_augment()

    def __len__(self) -> int:
        return len(self.folds)

    def __getitem__(self, idx: int):
        fold = self.root / self.folds[idx]
        audio_path = fold / "audio.wav"
        image_path = fold / "image.jpg"
        meta = json.loads((fold / "meta.json").read_text())
        label = float(meta["brix"])

        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        audio = self.audio_transform(audio)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)

        return audio_tensor, image_tensor, label
