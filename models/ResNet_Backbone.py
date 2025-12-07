import torch
from torch import nn
import torchvision.models as models


class ResNetBackbone(torch.nn.Module):
    """
    x: Input tensor of shape (B, C, H, W)

    Returns:
    A tuple of feature maps from different layers:
        - x1 : Output after layer1 (B, 256, H/4, W/4)
        - x2 : Output after layer2 (B, 512, H/8, W/8)
        - x3 : Output after layer3 (B, 1024, H/16, W/16)
        - out: Output of resnet backbone (B, 1000)
    """
    def __init__(self, in_channels=1):
        super(ResNetBackbone, self).__init__()
        if in_channels != 3:
            self.fix_channels = nn.Conv2d(in_channels, 3, kernel_size=1)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.pre_layers = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.post_layers = nn.Sequential(
            resnet.avgpool,
            nn.Flatten(),
            resnet.fc
        )

    def forward(self, x):
        if hasattr(self, 'fix_channels'):
            x = self.fix_channels(x)
        x = self.pre_layers(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.post_layers(x4)
        return x1, x2, x3, x


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
        layer1_out, layer2_out, layer3_out, batch_out  = model(img_frame)
        print(f"Layer1 output shape: {layer1_out.shape}")
        assert layer1_out.shape == (img_frame.shape[0], 256, H // 4, W // 4)
        print(f"Layer2 output shape: {layer2_out.shape}")
        assert layer2_out.shape == (img_frame.shape[0], 512, H // 8, W // 8)
        print(f"Layer3 output shape: {layer3_out.shape}")
        assert layer3_out.shape == (img_frame.shape[0], 1024, H // 16, W // 16)
        print(f"Batch output shape: {batch_out.shape}")
        assert batch_out.shape == (img_frame.shape[0], 1000)

    print(f"Resnet Encoder Model size: {get_model_size_mb(model):.3f} MB")
