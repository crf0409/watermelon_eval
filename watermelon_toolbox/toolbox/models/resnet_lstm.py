from .base import BaseModel, register_model


@register_model("resnet_lstm")
class ResNetLSTM(BaseModel):
    def __init__(self):
        super().__init__()

    def forward(self, audio, image):
        raise NotImplementedError
