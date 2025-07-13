#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import torch

from toolbox.models import get_model


def main(model_path: Path, out: Path) -> None:
    model = get_model("resnet_lstm")
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    dummy_audio = torch.randn(1, 1, 48000)
    dummy_img = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, (dummy_audio, dummy_img), out)
    print(f"Exported to {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.model, args.output)
