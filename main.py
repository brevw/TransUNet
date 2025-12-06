import argparse
import torch

TRAINED_MODEL_PATH = "trained_model.pth"

def main():

    # use accelerator if available
    DEVICE = None
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    parser = argparse.ArgumentParser(description="CLI tool.")
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing step.", default=False)
    parser.add_argument("--train", action="store_true", help="Run training step.", default=False)
    args = parser.parse_args()

    if args.preprocess:
        from scripts.preprocess import preprocess_data
        preprocess_data()
        print("Preprocessing completed.")

    if args.train:
        from scripts.preprocess import load_test_names_and_sizes
        from scripts.dataloaders import TrainDataset, TestDataset
        from torch.utils.data import DataLoader, Dataset
        BATCH_SIZE = 8

        names, original_sizes = load_test_names_and_sizes()
        train_dataset = TrainDataset()
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataset = TestDataset()
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)




if __name__ == "__main__":
    main()
