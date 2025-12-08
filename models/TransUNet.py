import torch
from torch import nn
from models.ResNet_Backbone import ResNetBackbone
from models.ViT import ViT

class Decode_Block(torch.nn.Module):
    """
    x: Input tensor of shape (B, C, H/(2*n), W/(2*n))
    skip_connection: Tensor from the encoder of shape (B, C_skip, H/n, W/n)
    Returns:
        Output tensor of shape (B, out_channels, H/n, W/n)
    """
    def __init__(self, in_channels: int, skip_channels, out_channels: int):
        super(Decode_Block, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, skip_connection):
        x = self.upconv(x) # shape (B, out_channels, H/n, W/n)
        x = torch.cat((x, skip_connection), dim=1) # shape (B, out_channels + C_skip, H/n, W/n)
        x = self.conv_block(x) # shape (B, out_channels, H/n, W/n)
        return x


class TransUNet(torch.nn.Module):
    def __init__(
        self,
        img_size: int = 512,
        in_channels: int = 1,
        out_channels: int = 1,
        # ViT Settings
        vit_depth: int = 12,
        vit_heads: int = 12,
        vit_dim_head: int = 128,
        vit_mlp_dim: int = 3072,
        vit_dropout: float = 0.1,
        # The hidden dimension inside the ViT
        vit_embed_dim: int = 768
    ):
        super(TransUNet, self).__init__()

        # Backbone Encoder
        self.encoder = ResNetBackbone(in_channels=in_channels)
        resnet_out_channels = 256  # Output channels from ResNet Backbone

        # ViT Bottleneck
        patch_size = 2
        self.vit = ViT(
            image_size=img_size,
            patch_size= patch_size,
            dim_in=vit_embed_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim,
            channels=resnet_out_channels,
            dim_head=vit_dim_head,
            dropout=vit_dropout
        )

        self.vit_conv = nn.Sequential(
            nn.Conv2d(vit_embed_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder Blocks
        self.decode3 = Decode_Block(in_channels=512, skip_channels=256, out_channels=256)
        self.decode2 = Decode_Block(in_channels=256, skip_channels=128, out_channels=128)
        self.decode1 = Decode_Block(in_channels=128, skip_channels=64, out_channels=64)
        self.upconv0 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # Encoder
        skip1, skip2, skip3 = self.encoder(x)
        # skip1: (B, 64, H/2, W/2)
        # skip2: (B, 128, H/4, W/4)
        # skip3: (B, 256, H/8, W/8)

        # Bottleneck with ViT
        vit_output = self.vit(skip3)
        vit_output = vit_output[:, 1:, :] # Remove cls token
                                          # shape (B, N, vit_embed_dim)
        B, N, D = vit_output.shape
        N_sqrt = int((N) ** 0.5)
        vit_output = vit_output.permute(0, 2, 1).reshape(B, D, N_sqrt, N_sqrt) # shape (B, vit_embed_dim, H/16, W/16)
        vit_output = self.vit_conv(vit_output) # shape (B, 512, H/16, W/16)

        # Decoder
        x = self.decode3(vit_output, skip3) # shape (B, 256, H/8, W/8)
        x = self.decode2(x, skip2) # shape (B, 128, H/4, W/4)
        x = self.decode1(x, skip1) # shape (B, 64, H/2, W/2)
        x = self.upconv0(x) # shape (B, 16, H, W)
        x = self.final_conv(x) # shape (B, out_channels, H, W)
        return x




if __name__ == "__main__":
    from utils.utils import get_model_size_mb
    H, W = 512, 512
    in_channels = 512
    out_channels = 256

    model = TransUNet(in_channels=1, out_channels=1)
    input_tensor = torch.randn((2, 1, 512, 512))
    print(f"input tensor shape: {input_tensor.shape}")
    output_tensor = model(input_tensor)
    print(f"output tensor shape: {output_tensor.shape}")
    print(f"Trans-UNet Model size: {get_model_size_mb(model):.3f} MB")
