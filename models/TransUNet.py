import torch
from torch import nn


class TransUNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        ...

    def forward(self, x):
        ...


if __name__ == "__main__":
    from utils.utils import get_model_size_mb
    model = TransUNet(in_channels=1, out_channels=1)
    print(f"Trans-UNet Model size: {get_model_size_mb(model):.3f} MB")
