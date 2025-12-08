import argparse
import torch
from utils.utils import MODEL_UNET, MODEL_TRANS_UNET, MODEL_UNET_PATH, MODEL_TRANS_UNET_PATH, RESULTS_PATH
import logging
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import pandas as pd

def save_video_sequence(video_frames, output_path, fps=10):
    if not video_frames:
        return
    first_frame = video_frames[0]
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height), # OpenCV expects (Width, Height)
        isColor=False
    )
    for frame in video_frames:
        video_writer.write(frame)
    video_writer.release()

def process_submission_volume(video_frames, last_name, global_id_counter, ids, values):
    if len(video_frames) == 0:
        return global_id_counter

    video_volume = np.stack(video_frames, axis=0)
    video_volume = video_volume.transpose(1, 2, 0)

    arr = video_volume.flatten()
    from utils.utils import get_sequences # Ensure this is imported
    starts, lengths = get_sequences(arr)

    for s, l in zip(starts, lengths):
        ids.append(f"{last_name}_{global_id_counter}")
        values.append([int(s), int(l)])
        global_id_counter += 1
    return global_id_counter



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
    parser.add_argument("--video_mode", action="store_true", help="Output in video mode.", default=False)
    parser.add_argument("--submit_mode", action="store_true", help="Output in submission mode.", default=False)
    parser.add_argument("--limit_output", type=int, help="Limit output to first N samples.", default=100)
    args = parser.parse_args()

    if args.preprocess:
        from scripts.preprocess import preprocess_data
        # Preprocessing step
        logging.info("Starting preprocessing...")
        preprocess_data()
        logging.info("Preprocessing completed.")

    else:
        from models.UNet import UNet
        from models.TransUNet import TransUNet
        from scripts.preprocess import load_test_names_and_sizes
        from scripts.dataset import TrainDataset, TestDataset
        from torch.utils.data import DataLoader

        # Initialize model
        model: torch.nn.Module
        path_to_model_weights: str

        if args.model == MODEL_UNET:
            model = UNet(in_channels=1, out_channels=1).to(DEVICE)
            path_to_model_weights = MODEL_UNET_PATH
        elif args.model == MODEL_TRANS_UNET:
            model = TransUNet(in_channels=1, out_channels=1).to(DEVICE)
            path_to_model_weights = MODEL_TRANS_UNET_PATH
        else:
            raise ValueError(f"Model {args.model} not recognized.")

        if args.train:
            # Training step
            logging.info("Starting training...")
            BATCH_SIZE = 8
            LR = 1e-4
            EPOCHS = 60

            train_loader = DataLoader(TrainDataset(), batch_size=BATCH_SIZE, shuffle=True)
            from models.loss import DiceBCELoss

            criterion = DiceBCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            for epoch in range(EPOCHS):
                model.train()
                epoch_loss = 0.0
                for imgs, masks in tqdm(train_loader):
                    imgs = imgs.to(DEVICE).float()
                    masks = masks.to(DEVICE).float()

                    outputs = model(imgs)
                    loss = criterion(outputs, masks)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print("finished step")

                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(train_loader)
                logging.info(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

            # Save model weights
            torch.save(model.state_dict(), path_to_model_weights)
            logging.info(f"Model weights saved to {path_to_model_weights}.")

            logging.info("Training completed.")

        elif args.inference:
            from utils.utils import get_sequences

            test_loader = DataLoader(TestDataset(), batch_size=1, shuffle=False)
            names, original_sizes = load_test_names_and_sizes()
            test_dataset_size = len(test_loader)
            # threshold for mask binarization
            THRESHOLD = 0.6

            # Load model weights
            logging.info(f"Loading model weights from {args.model}...")
            state_dict = torch.load(path_to_model_weights, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)

            # logging the mode
            if args.video_mode:
                logging.info("Inference in video mode.")
            elif args.submit_mode:
                logging.info("Inference in submission mode.")
            else:
                logging.info("Inference in image mode.")

            # Inference loop
            logging.info("Starting inference...")
            model.eval()

            ids = []     # used in submit
            values = []  # mode

            video_frames = []  # used in video and submit
            last_name = None   # mode
            global_id_counter = 0

            limit_output = 0 # limit number of outputs
            idx = 0
            for img in tqdm(test_loader):
                name = names[idx]
                original_size = original_sizes[idx]
                pred_mask = None
                with torch.no_grad():
                    img = img.to(DEVICE)
                    output = model(img)
                    pred_mask = (torch.sigmoid(output) > THRESHOLD).float().cpu().numpy()
                    # Resize mask to original size
                    pred_mask = torch.nn.functional.interpolate(torch.from_numpy(pred_mask),
                      size=original_size,
                      mode='nearest').squeeze(0).squeeze(0).numpy()
                    img_resize = torch.nn.functional.interpolate(img.cpu(),
                        size=original_size,
                        mode='bilinear',
                        align_corners=False).squeeze(0).squeeze(0).numpy()
                    img_resize_uint8 = (img_resize * 255).astype('uint8')
                    pred_mask_uint8 = (pred_mask * 255).astype('uint8')
                    combined = np.hstack((img_resize_uint8, pred_mask_uint8))
                    # Save predicted mask
                    if args.video_mode:
                        if last_name is not None and last_name != name:
                            video_path = f"{RESULTS_PATH}{last_name}.mp4"
                            save_video_sequence(video_frames, video_path, fps=10)
                            limit_output += 1
                            if limit_output >= args.limit_output:
                                logging.info(f"Reached limit of {args.limit_output} outputs. Exiting inference.")
                                video_frames = []
                                break
                            video_frames = []
                        video_frames.append(combined)
                        last_name = name
                    elif args.submit_mode:
                        if last_name is not None and last_name != name:
                            global_id_counter = process_submission_volume(video_frames, last_name, global_id_counter, ids, values)
                            video_frames = []
                        video_frames.append(pred_mask.astype(np.int32))
                        last_name = name
                    else:
                        pred_mask_img = Image.fromarray((pred_mask * 255).astype('uint8'))
                        pred_mask_img.save(f"{RESULTS_PATH}{idx}.png")
                        limit_output += 1
                        if limit_output >= args.limit_output:
                            logging.info(f"Reached limit of {args.limit_output} outputs. Exiting inference.")
                            break
                idx += 1

            if args.video_mode and len(video_frames) > 0:
                    video_path = f"{RESULTS_PATH}{last_name}.mp4"
                    save_video_sequence(video_frames, video_path, fps=10)
            elif args.submit_mode and len(video_frames) > 0:
                global_id_counter = process_submission_volume(video_frames, last_name, global_id_counter, ids, values)
                df = pd.DataFrame({"id":ids, "value":[list(map(int, minili)) for minili in values]})
                df.to_csv(f"{RESULTS_PATH}submission_{THRESHOLD}.csv", index=False)

            logging.info("Inference completed.")


if __name__ == "__main__":
    main()
