import torch
from torch import nn
import torch.nn.functional as F

class IoULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (Batch, 1, H, W) raw logits
        # targets: (Batch, 1, H, W) binary 0 or 1
        probs = torch.sigmoid(inputs)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        total = (probs + targets).sum()
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou
class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        if targets.max() > 1.0:
            targets = targets / 255.0
        intersection = (inputs * targets).sum()
        dice_score = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice_score
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return dice_loss + bce_loss
