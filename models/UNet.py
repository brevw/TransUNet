import torch
from torch import nn

class DoubleConv(torch.nn.Module):
    # input:  (B, in_channels , H, W)
    # output: (B, out_channels, H, W)
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.seq(x)

class UNet(torch.nn.Module):
    # input:  (B, in_channels , H, W)
    # output: (B, out_channels, H, W)
    def __init__(self, in_channels: int, out_channels: int):
        super(UNet, self).__init__()

        # Encode (downsample)
        self.increase = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(256, 512))

        # Bottleneck
        self.down4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(512, 1024))

        # Decode (upsample)
        self.up4 = nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2))
        self.up_conv4 = DoubleConv(1024, 512)
        self.up3 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))
        self.up_conv3 = DoubleConv(512, 256)
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2))
        self.up_conv2 = DoubleConv(256, 128)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))
        self.up_conv1 = DoubleConv(128, 64)

        # Output
        self.decrease = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Go towards Bottleneck
        x0 = self.increase(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.down4(x3)

        # Upsample from the Bottleneck
        x = self.up4(x)
        x = self.up_conv4(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.up_conv3(torch.cat([x, x2], dim=1))
        x = self.up2(x)
        x = self.up_conv2(torch.cat([x, x1], dim=1))
        x = self.up1(x)
        x = self.up_conv1(torch.cat([x, x0], dim=1))
        x = self.decrease(x)
        return x
