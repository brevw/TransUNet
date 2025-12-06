import numpy as np
from tqdm import tqdm
from utils.utils import load_zipped_pickle, save_zipped_pickle, \
    COMPRESSED_TRAIN_DATA, COMPRESSED_TEST_DATA, DECOMPRESSED_TRAIN_PATH, DECOMPRESSED_TEST_PATH
from PIL import Image



def process_train_data(data):
    i = 0
    for item in tqdm(data):
        video = item['video']
        name = item['name']
        height, width, n_frames = video.shape
        mask = np.zeros((height, width, n_frames), dtype=bool)
        for frame in item['frames']:
            mask[:, :, frame] = item['label'][:, :, frame]
            video_frame = video[:, :, frame]
            mask_frame = mask[:, :, frame]
            video_frame = np.expand_dims(video_frame, axis=2).astype(np.float32)
            mask_frame = np.expand_dims(mask_frame, axis=2).astype(np.int32)

            #save files
            img_frame = video_frame.squeeze()
            img_frame = Image.fromarray(img_frame.astype(np.uint8))
            img_frame = img_frame.resize((512, 512))
            img_frame.save(f"{DECOMPRESSED_TRAIN_PATH}{i}_frame.png")

            img_mask = mask_frame.squeeze() * 255
            img_mask = Image.fromarray(img_mask.astype(np.uint8))
            img_mask = img_mask.resize((512, 512))
            img_mask.save(f"{DECOMPRESSED_TRAIN_PATH}{i}_mask.png")

            i += 1

def process_test_data(data):
    names = []
    original_sizes = []
    i = 0
    for item in tqdm(data):
        video = item['video']
        video = video.astype(np.float32).transpose((2, 0, 1))
        video = np.expand_dims(video, axis=3)
        names += [item['name'] for _ in video]

        #save files
        for j in range(video.shape[0]):
            img_frame = video[j].squeeze()
            original_sizes.append(img_frame.shape)
            img_frame = Image.fromarray(img_frame.astype(np.uint8))
            img_frame = img_frame.resize((512, 512))
            img_frame.save(f"{DECOMPRESSED_TEST_PATH}{i}.png")
            i += 1
    return names, original_sizes


def preprocess_data():
    train_data_unpacked = load_zipped_pickle(COMPRESSED_TRAIN_DATA)
    test_data_unpacked = load_zipped_pickle(COMPRESSED_TEST_DATA)
    process_train_data(train_data_unpacked)
    names, original_sizes = process_test_data(test_data_unpacked)
    # save names and original sizes
    save_zipped_pickle((names, original_sizes), f"{DECOMPRESSED_TEST_PATH}test_names_sizes.pkl")

def load_test_names_and_sizes():
    names, original_sizes = load_zipped_pickle(f"{DECOMPRESSED_TEST_PATH}/test_names_sizes.pkl")
    return names, original_sizes
