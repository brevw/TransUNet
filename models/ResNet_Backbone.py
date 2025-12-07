import torch
from torch import nn

class Block(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Block, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x) # shape (B, out_channels, H/2, W/2)
        out = self.conv_block(x) # shape (B, out_channels, H/2, W/2)
        out += identity # shape (B, out_channels, H/2, W/2)
        out = self.relu(out)
        return out # shape (B, out_channels, H/2, W/2)


class ResNetBackbone(torch.nn.Module):
    """
    x: Input tensor of shape (B, C, H, W)

    Returns:
    A tuple of feature maps from different layers:
        - x1 : Output after layer1 (B, 64, H/2, W/2)
        - x2 : Output after layer2 (B, 128, H/4, W/8)
        - x3 : Output after layer3 (B, 256, H/8, W/8)
    """
    def __init__(self, in_channels=1):
        super(ResNetBackbone, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = Block(32, 64)
        self.layer2 = Block(64, 128)
        self.layer3 = Block(128, 256)

    def forward(self, x):
        x = self.initial_conv(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x1, x2, x3


if __name__ == "__main__":
    from utils.utils import get_model_size_mb
    from scripts.dataset import TrainDataset
    from torch.utils.data import DataLoader

    train_dataset = TrainDataset()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    img_frame, img_mask = next(iter(train_loader))
    print(f"Image frames batch shape: {img_frame.shape}")
    print(f"Image masks batch shape: {img_mask.shape}")
    H, W = img_frame.shape[2], img_frame.shape[3]

    model = ResNetBackbone()
    model.eval()
    with torch.no_grad():
        layer1_out, layer2_out, layer3_out= model(img_frame)
        print(f"Layer1 output shape: {layer1_out.shape}")
        print(f"Layer2 output shape: {layer2_out.shape}")
        print(f"Layer3 output shape: {layer3_out.shape}")

    print(f"Resnet Encoder Model size: {get_model_size_mb(model):.3f} MB")
