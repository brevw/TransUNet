import torch
from torch import nn

class Block(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(Block, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv_block(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetBackbone(torch.nn.Module):
    """
    x: Input tensor of shape (B, C, H, W)

    Returns:
        - x1 : (B, 64,  H/2, W/2)
        - x2 : (B, 128, H/4, W/4)
        - x3 : (B, 256, H/8, W/8)
    """
    def __init__(self, in_channels=1):
        super(ResNetBackbone, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = Block(32, 64, stride=1) # shape H/2
        self.layer2 = Block(64, 128, stride=2) # shape H/4
        self.layer3 = Block(128, 256, stride=2) # shape H/8

    def forward(self, x):
        # Input: (B, C, H, W)

        x = self.initial_conv(x) # -> (B, 32, H/2, H/2)

        x1 = self.layer1(x)      # -> (B, 64, H/2, H/2)
        x2 = self.layer2(x1)     # -> (B, 128, H/4, H/4)
        x3 = self.layer3(x2)     # -> (B, 256, H/8, H/8)

        return x1, x2, x3

# Validation
if __name__ == "__main__":
    model = ResNetBackbone(in_channels=3)
    dummy_input = torch.randn(1, 3, 512, 512)
    x1, x2, x3 = model(dummy_input)

    print(f"Input: {dummy_input.shape}")
    print(f"x1 (Target H/2): {x1.shape}") # Should be 256
    print(f"x2 (Target H/4): {x2.shape}") # Should be 128
    print(f"x3 (Target H/8): {x3.shape}") # Should be 64
