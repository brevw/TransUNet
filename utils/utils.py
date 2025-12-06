import gzip
import pickle
import numpy as np
import os

# Models
MODEL_UNET = "unet"
MODEL_TRANS_UNET = "trans-unet"
MODEL_UNET_PATH = "models/unet_model.pth"
MODEL_TRANS_UNET_PATH = "models/trans_unet_model.pth"


#Paths Dataset
COMPRESSED_TRAIN_DATA = "data_compressed/train.pkl"
COMPRESSED_TEST_DATA = "data_compressed/test.pkl"

DECOMPRESSED_TRAIN_PATH = "data/train/"
DECOMPRESSED_TEST_PATH = "data/test/"

def load_zipped_pickle(filename: str):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def save_zipped_pickle(obj, filename: str):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

def get_train_dataset_size():
    if os.path.exists(DECOMPRESSED_TRAIN_PATH):
        names = [name for name in os.listdir(DECOMPRESSED_TRAIN_PATH) if os.path.isfile(os.path.join(DECOMPRESSED_TRAIN_PATH, name)) and name.endswith('.png')]
        return len(names) // 2
    else:
        raise FileNotFoundError(f"Directory {DECOMPRESSED_TRAIN_PATH} does not exist.")


def get_sequences(arr):
    first_indices, last_indices, lengths = [], [], []
    n, i = len(arr), 0
    arr = [0] + list(arr) + [0]
    for index, value in enumerate(arr[:-1]):
        if arr[index+1]-arr[index] == 1:
            first_indices.append(index)
        if arr[index+1]-arr[index] == -1:
            last_indices.append(index)
    lengths = list(np.array(last_indices)-np.array(first_indices))
    return first_indices, lengths
