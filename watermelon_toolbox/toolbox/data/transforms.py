from __future__ import annotations

import random
from typing import Tuple

import librosa
import numpy as np
import torch
import torchvision.transforms as T


class RandomCropAudio:
    def __init__(self, length: int = 48000):
        self.length = length

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) <= self.length:
            if len(audio) < self.length:
                pad = self.length - len(audio)
                audio = np.pad(audio, (0, pad))
            return audio
        start = random.randint(0, len(audio) - self.length)
        return audio[start : start + self.length]


class AddNoise:
    def __init__(self, snr: float = 20.0):
        self.snr = snr

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(audio**2))
        noise_rms = rms / (10 ** (self.snr / 20))
        noise = np.random.normal(0, noise_rms, audio.shape)
        return audio + noise


class SpecAugment:
    def __init__(self, time_mask: int = 10, freq_mask: int = 8):
        self.time_mask = time_mask
        self.freq_mask = freq_mask

    def __call__(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        spec_db = librosa.power_to_db(spec)
        spec = torch.tensor(spec_db)
        t = spec.size(1)
        f = spec.size(0)
        t0 = random.randint(0, max(0, t - self.time_mask))
        spec[:, t0 : t0 + self.time_mask] = 0
        f0 = random.randint(0, max(0, f - self.freq_mask))
        spec[f0 : f0 + self.freq_mask, :] = 0
        return spec.numpy()


def image_augment() -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(224),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomHorizontalFlip(),
        ]
    )
