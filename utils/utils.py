import gzip
import pickle
import numpy as np

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
