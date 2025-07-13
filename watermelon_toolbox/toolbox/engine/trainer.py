from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        work_dir: Path = Path("runs"),
        max_epochs: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.device = device
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.work_dir)
        self.best_loss = float("inf")

    def _run_epoch(self, epoch: int) -> float:
        self.model.train()
        running = 0.0
        pbar = tqdm(self.train_loader, desc=f"Train {epoch}")
        for audio, image, label in pbar:
            audio = audio.to(self.device)
            image = image.to(self.device)
            label = torch.tensor(label, dtype=torch.float32, device=self.device).unsqueeze(1)
            self.optimizer.zero_grad()
            pred = self.model(audio, image)
            loss = self.criterion(pred, label)
            loss.backward()
            self.optimizer.step()
            running += loss.item() * audio.size(0)
        return running / len(self.train_loader.dataset)

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        if not self.val_loader:
            return 0.0
        self.model.eval()
        running = 0.0
        for audio, image, label in tqdm(self.val_loader, desc=f"Val {epoch}"):
            audio = audio.to(self.device)
            image = image.to(self.device)
            label = torch.tensor(label, dtype=torch.float32, device=self.device).unsqueeze(1)
            pred = self.model(audio, image)
            loss = self.criterion(pred, label)
            running += loss.item() * audio.size(0)
        return running / len(self.val_loader.dataset)

    def fit(self):
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._run_epoch(epoch)
            val_loss = self._validate(epoch)
            self.writer.add_scalar("loss/train", train_loss, epoch)
            if self.val_loader:
                self.writer.add_scalar("loss/val", val_loss, epoch)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), self.work_dir / "best.ckpt")
