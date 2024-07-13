import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math
from scipy.signal import resample
from scipy.stats import mode


####################################################################
# Downsampling the majority class
def downsample_data(data, labels):
    
    # Ensure labels are 1D by taking the mode along the time axis
    labels = mode(labels, axis=1).mode.flatten()
    
    print(f"Initial data shape: {data.shape}, labels shape: {labels.shape}")

        
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Counts per class before downsampling: {dict(zip(unique, counts))}")
    
    minority_class = unique[np.argmin(counts)]
    majority_class = unique[np.argmax(counts)]
    
    minority_class_data = data[labels == minority_class]
    majority_class_data = data[labels == majority_class]
    
    print(f"Minority class size: {len(minority_class_data)}")
    print(f"Majority class size: {len(majority_class_data)}")
    num_to_select = len(minority_class_data)
    selected_indices = np.random.choice(len(majority_class_data), num_to_select, replace=False)
    downsampled_majority_class_data = majority_class_data[selected_indices]
    
    downsampled_data = np.concatenate((minority_class_data, downsampled_majority_class_data), axis=0)
    downsampled_labels = np.array([minority_class] * len(minority_class_data) + [majority_class] * num_to_select)
    
    indices = np.arange(len(downsampled_labels))
    np.random.shuffle(indices)
    
    unique, counts = np.unique(downsampled_labels, return_counts=True)
    print("Class Distribution after downsampling:", dict(zip(unique, counts)))
    print(f"Downsampled data shape: {downsampled_data.shape}, Downsampled labels shape: {downsampled_labels.shape}")
    
    return downsampled_data[indices], downsampled_labels[indices]
############################################################################
def load_folds_data_shhs(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    # Shuffle the files
    np.random.shuffle(files)
    folds_data = {}
    
    all_data = []
    all_labels = []
    
    # Load and concatenate data from all files
    for file_path in files:
        if not os.path.exists(file_path):
            print(f"Error: File does not exist {file_path}")
            continue
        
        try:
            data = np.load(file_path)
            x_data = data['x']
            y_data = data['y']
            
            all_data.append(x_data)
            all_labels.append(y_data)
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            continue
        
    # Concatenate all data and labels
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Ensure all_labels is 2D
    if all_labels.ndim == 1:
        all_labels = np.expand_dims(all_labels, axis=-1)
    
    total_samples = len(all_data)
    
    
    print(f"Total samples from all files: {total_samples}")
    
    train_samples = int(0.7 * total_samples)
    
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    train_indices = indices[:train_samples]
    test_indices = indices[train_samples:]
    
    train_data = all_data[train_indices]
    train_labels = all_labels[train_indices]
    test_data = all_data[test_indices]
    test_labels = all_labels[test_indices]
    
    print(f"Training set shape before downsampling: {train_labels.shape}")
    print(f"Testing set shape: {test_labels.shape}")
    
    # Perform downsampling on training data
    train_data, train_labels = downsample_data(train_data, train_labels)
    
    # Save data to new file paths
    train_file_path = os.path.join(np_data_path, "train_data_shhs.npz")
    test_file_path = os.path.join(np_data_path, "test_data_shhs.npz")
    
    np.savez(train_file_path, x=train_data, y=train_labels)
    np.savez(test_file_path, x=test_data, y=test_labels)
    
    print(f"Train file path: {train_file_path}")
    print(f"Test file path: {test_file_path}")
    
    # fold_id = 0
    folds_data[0] = [train_file_path, test_file_path]
    
    return folds_data



def calc_class_weight(labels_count):
    # Already applied downsampling
    num_classes = len(labels_count)
    class_weight = [1.0] * num_classes
    # class_weight = [1.5, 1.0]
    # print(f"Number of Classes: {num_classes}")
    # print(f"Class weighting: {class_weight}")

    return class_weight


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)