from torch import nn


class BaseModel(nn.Module):
    """Base model interface."""

    def forward(self, audio, image):
        raise NotImplementedError


_REGISTRY = {}


def register_model(name):
    def wrapper(cls):
        _REGISTRY[name] = cls
        return cls

    return wrapper


def get_model(name, *args, **kwargs):
    return _REGISTRY[name](*args, **kwargs)
