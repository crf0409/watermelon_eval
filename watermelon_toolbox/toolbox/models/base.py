from __future__ import annotations

from typing import Callable, Dict, Type

import torch
from torch import nn


class BaseModel(nn.Module):
    def forward(self, audio: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name: str) -> Callable[[Type[BaseModel]], Type[BaseModel]]:
    def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **kwargs) -> BaseModel:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name}")
    return _MODEL_REGISTRY[name](**kwargs)
