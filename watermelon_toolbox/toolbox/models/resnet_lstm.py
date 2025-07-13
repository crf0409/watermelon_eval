from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from .base import BaseModel, register_model


@register_model("resnet_lstm")
class ResNetLSTM(BaseModel):
    def __init__(self, lstm_hidden: int = 128):
        super().__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()
        self.lstm = nn.LSTM(input_size=100, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden + 2048, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, audio: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # audio: (B, 1, 48000)
        b, _, n = audio.shape
        audio = audio.view(b, 160, 300)  # (B, 160, 300)
        _, (h, _) = self.lstm(audio)
        h = h.squeeze(0)
        x = self.resnet(image)
        out = torch.cat([h, x], dim=1)
        out = self.fc(out)
        return out
