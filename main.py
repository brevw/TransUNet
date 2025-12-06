import argparse
import torch
from utils.utils import MODEL_UNET, MODEL_TRANS_UNET, MODEL_UNET_PATH, MODEL_TRANS_UNET_PATH
import logging
from tqdm import tqdm

def main():
    # set logging format
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # use accelerator if available
    DEVICE = None
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    logging.info(f"Using device: {DEVICE}")

    parser = argparse.ArgumentParser(description="CLI tool.")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing step.", default=False)
    parser.add_argument("--train", action="store_true", help="Run training step.", default=False)
    parser.add_argument("--inference", action="store_true", help="Run inference step.", default=False)
    parser.add_argument("--model", type=str, help="Path to the model file.", default=MODEL_UNET)
    args = parser.parse_args()

    if args.preprocess:
        from scripts.preprocess import preprocess_data
        logging.info("Starting preprocessing...")
        preprocess_data()
        logging.info("Preprocessing completed.")

    else:
        from models.UNet import UNet
        from models.TransUNet import TransUNet
        from scripts.preprocess import load_test_names_and_sizes
        from scripts.dataset import TrainDataset, TestDataset
        from torch.utils.data import DataLoader

        model: torch.nn.Module
        if args.model == MODEL_UNET:
            model = UNet(in_channels=1, out_channels=1).to(DEVICE)
        elif args.model == MODEL_TRANS_UNET:
            model = TransUNet(in_channels=1, out_channels=1).to(DEVICE)
        else:
            raise ValueError(f"Model {args.model} not recognized.")

        if args.train:
            logging.info("Starting training...")
            BATCH_SIZE = 8
            LR = 1e-4
            EPOCHS = 60

            names, original_sizes = load_test_names_and_sizes()
            train_loader = DataLoader(TrainDataset(), batch_size=BATCH_SIZE, shuffle=True)
            from models.loss import DiceBCELoss

            criterion = DiceBCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            for epoch in range(EPOCHS):
                model.train()
                epoch_loss = 0.0
                for imgs, masks in tqdm(train_loader):
                    imgs = imgs.to(DEVICE)
                    masks = masks.to(DEVICE)

                    outputs = model(imgs)
                    loss = criterion(outputs, masks)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(train_loader)
                logging.info(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")



            logging.info("Training completed.")

        elif args.inference:
            logging.info("Starting inference...")
            test_loader = DataLoader(TestDataset(), batch_size=1, shuffle=False)


            logging.info("Inference completed.")




if __name__ == "__main__":
    main()
