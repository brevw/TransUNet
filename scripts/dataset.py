from torch.utils.data import DataLoader, Dataset
from utils.utils import get_train_dataset_size, \
    DECOMPRESSED_TRAIN_PATH, DECOMPRESSED_TEST_PATH
from scripts.preprocess import load_test_names_and_sizes
from PIL import Image
import numpy as np
import torch
from torchvision import transforms


augment_transform = transforms.RandomAffine(
    degrees=(-15, 15),       # Random rotation between -15 and +15 degrees
    translate=(0.1, 0.1),    # Random horizontal/vertical shift up to 10%
    scale=(0.9, 1.1),        # Random scaling between 90% and 110%
    shear=None,              # No shearing applied
    interpolation=transforms.InterpolationMode.NEAREST, # IMPORTANT for masks
    fill=0                   # Fill background with 0 (black)
)

class TrainDataset(Dataset):
    """
    Dataset class for training data.
    Output will have the following shape:
        - frame -> (B, 1, H, W)
        - mask  -> (B, 1, H, W)
    """
    def __init__(self):
        self.size = get_train_dataset_size()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_frame = Image.open(f"{DECOMPRESSED_TRAIN_PATH}{idx}_frame.png").convert("L")
        img_frame = torch.from_numpy(np.array(img_frame)).unsqueeze(0)
        img_mask = Image.open(f"{DECOMPRESSED_TRAIN_PATH}{idx}_mask.png").convert("L")
        img_mask = torch.from_numpy(np.array(img_mask)).unsqueeze(0)

        # Augmentation by adding random translation and rotation
        img_frame = augment_transform(img_frame)
        img_mask = augment_transform(img_mask)

        # Normalize images so that values are between 0 and 1
        img_frame = img_frame.float() / 255.0
        img_mask = img_mask.type(torch.int32) // 255

        return img_frame, img_mask

class TestDataset(Dataset):
    """
    Dataset class for testing data.
    Output will have the following shape:
        - frame -> (B, 1, H, W)
    """

    def __init__(self):
        self.size = len(load_test_names_and_sizes()[0])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_frame = Image.open(f"{DECOMPRESSED_TEST_PATH}{idx}.png").convert("L")
        img_frame = torch.from_numpy(np.array(img_frame)).unsqueeze(0)

        # Normalize images so that values are between 0 and 1
        img_frame = img_frame.float() / 255.0

        return img_frame

# testing the TrainDataset [code snippet]
if __name__ == "__main__":
    from torchvision import utils as vutil
    TEST_TRAIN_DATASET = True

    if TEST_TRAIN_DATASET:
        train_dataset = TrainDataset()
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        img_frames, img_masks = next(iter(train_loader))
        print(f"Image frames batch shape: {img_frames.shape}")
        print(f"Image masks batch shape: {img_masks.shape}")
        from matplotlib import pyplot as plt
        #plot both images and masks
        plt.figure(figsize=(10, 7)) # Increased size for visibility
        plt.axis('off')
        plt.title("Train Image Frames and Masks")

        combined_batch = torch.stack([img_frames, img_masks], dim=1).flatten(0, 1)
        grid_img = vutil.make_grid(combined_batch, nrow=8, padding=2, normalize=True)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
    else:
        test_dataset = TestDataset()
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        img_frames = next(iter(test_loader))
        print(f"Image frames batch shape: {img_frames.shape}")
        from matplotlib import pyplot as plt
        #plot image at index 0
        plt.figure(figsize=(10, 7))
        plt.axis('off')
        plt.title("Test Image Frames")
        grid_img = vutil.make_grid(img_frames, nrow=8, padding=2, normalize=True)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()


