#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from PIL import Image
import yaml

from toolbox.models import get_model
from toolbox.data.transforms import image_augment, RandomCropAudio
import librosa
import numpy as np


def load_audio(path: Path) -> torch.Tensor:
    audio, _ = librosa.load(path, sr=16000, mono=True)
    audio = RandomCropAudio()(audio)
    return torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return image_augment()(img).unsqueeze(0)


def main(model_path: Path, audio_file: Path, image_file: Path) -> None:
    model = get_model("resnet_lstm")
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    audio = load_audio(audio_file)
    image = load_image(image_file)
    with torch.no_grad():
        pred = model(audio, image)
    print(float(pred.item()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    main(args.model, args.audio, args.image)
