import argparse

TRAINED_MODEL_PATH = "trained_model.pth"

def main():
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
        names, original_sizes = load_test_names_and_sizes()
        print(len(names))
        print(len(original_sizes))


if __name__ == "__main__":
    main()
