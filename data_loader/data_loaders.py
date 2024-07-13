import torch
from torch.utils.data import Dataset
import os
import numpy as np
from scipy.stats import mode

class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy, self).__init__()

        # load files
        X_train = np.load(np_dataset[0])["x"]
        y_train = np.load(np_dataset[0])["y"]

        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)["x"]))
            y_train = np.append(y_train, np.load(np_file)["y"])

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator_np(training_files, subject_files, batch_size):
    # The paths are now strings instead of lists
    # Force the paths to be lists
    if not isinstance(training_files, list):
        training_files = [training_files]
    if not isinstance(subject_files, list):
        subject_files = [subject_files]
        
    train_dataset = LoadDataset_from_numpy(training_files)
    test_dataset = LoadDataset_from_numpy(subject_files)
    
    # Print shapes to debug
    print(f"Train dataset size: {train_dataset.len}, Test dataset size: {test_dataset.len}")
    print(f"Train dataset y_data shape: {train_dataset.y_data.shape}")
    print(f"Test dataset y_data shape: {test_dataset.y_data.shape}")
    
    # Ensure y_data arrays are 1D by taking the mode along the time axis
    if train_dataset.y_data.ndim > 1:
        train_dataset.y_data = mode(train_dataset.y_data, axis=1).mode.flatten()
    if test_dataset.y_data.ndim > 1:
        test_dataset.y_data = mode(test_dataset.y_data, axis=1).mode.flatten()

    # print(f"Flattened train dataset y_data shape: {train_dataset.y_data.shape}")
    # print(f"Flattened test dataset y_data shape: {test_dataset.y_data.shape}")

    # to calculate the ratio for the CAL
    all_ys = np.concatenate((train_dataset.y_data, test_dataset.y_data))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader, counts
