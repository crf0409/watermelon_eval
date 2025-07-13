from __future__ import annotations

import torch
from torch import nn
from tqdm import tqdm


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: str = "cpu") -> float:
    model.eval()
    model.to(device)
    total = 0.0
    count = 0
    with torch.no_grad():
        for audio, image, label in tqdm(loader, desc="Eval"):
            audio = audio.to(device)
            image = image.to(device)
            label = torch.tensor(label, dtype=torch.float32, device=device).unsqueeze(1)
            pred = model(audio, image)
            loss = nn.functional.mse_loss(pred, label, reduction="sum")
            total += loss.item()
            count += len(label)
    return total / count
