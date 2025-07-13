#!/usr/bin/env python
from __future__ import annotations

import yaml
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

from toolbox.data.dataset import WatermelonDataset
from toolbox.models import get_model
from toolbox.engine.trainer import Trainer


def main(config_path: Path) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg["data_root"])
    train_ds = WatermelonDataset(dataset_root, split="train")
    val_ds = WatermelonDataset(dataset_root, split="val")
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 4), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 4))

    model = get_model(cfg.get("model", "resnet_lstm"))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-4))
    criterion = nn.MSELoss()

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        work_dir=Path(cfg.get("work_dir", "runs/exp")),
        max_epochs=cfg.get("epochs", 10),
    )
    trainer.fit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    main(args.config)
