import os
import random
import numpy as np
import pickle

directory = 'data'

pkl_files = [file for file in os.listdir(directory) if file.endswith('.pkl')]

random.shuffle(pkl_files)

train_files = pkl_files[:50]
test_files = pkl_files[50:]

def load_arrays(file_list):
    arrays = []
    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'rb') as file:
            array = pickle.load(file)
            arrays.append(array)
    return arrays

train_arrays = load_arrays(train_files)
test_arrays = load_arrays(test_files)

train_array = np.concatenate(train_arrays)
test_array = np.concatenate(test_arrays)

np.save('train.npy', train_array)
np.save('test.npy', test_array)
