from keras.utils import Sequence
import numpy as np
import pandas as pd
from config import data_dir
import os
import cv2
import ipdb


class DataGenerator(Sequence):
    '''
    each call returns stack of X and ys, whers stack contains batch_size images randomly selected for datasets
    each frame will be square subframe of random size, but no larger than smallest dataset
    locations in frame will also be random
    todo: subframe size randomize // random rotation // random scaling
    '''

    def __init__(self, datasets, batch_size=8, subframe_size=(100, 100), normalize_subframes=False, epoch_size=1,
                 rotation=True, scaling=(1, 1)):
        # initialization

        self.datasets = datasets
        self.batch_size = batch_size
        self.subframe_size = subframe_size
        self.normalize_subframes = normalize_subframes
        self.epoch_size = epoch_size
        self.rotation = rotation
        self.scaling = scaling

        # load features and labels into DataFrame
        self.data = pd.DataFrame(index=datasets, columns=['X', 'y', 'corner_max'])
        for d in datasets:

            data_sub = np.load(os.path.join(data_dir, 'training_data', d + '.npz'))
            corner_max = (data_sub['X'].shape[0] - subframe_size[0],
                          data_sub['X'].shape[1] - subframe_size[1])  # subframe corner can be no further than corner_max
            self.data.loc[d, :] = (data_sub['X'], data_sub['y'], corner_max)

        self.shape_X = (batch_size,) + subframe_size + (self.data.loc[datasets[0], 'X'].shape[-1],)  # batch size X height X width X depth
        self.shape_y = (batch_size,) + subframe_size + (self.data.loc[datasets[0], 'y'].shape[-1],)

    def __getitem__(self, index):

        # gets data for batch
        batch_inds = np.random.randint(0, len(self.datasets), size=self.batch_size)  # indices of datasets to be used in batch
        X = np.zeros(self.shape_X)
        y = np.zeros(self.shape_y)

        for i, b in enumerate(batch_inds):
            corner = (np.random.randint(self.data.loc[self.datasets[b], 'corner_max'][0]),
                      np.random.randint(self.data.loc[self.datasets[b], 'corner_max'][1]))
            x_inds = slice(corner[0], corner[0]+self.subframe_size[0])
            y_inds = slice(corner[1], corner[1]+self.subframe_size[1])
            X[i] = self.data.loc[self.datasets[b], 'X'][x_inds, y_inds]
            y[i] = self.data.loc[self.datasets[b], 'y'][x_inds, y_inds]

        # rotate
        if self.rotation:
            rotations = np.random.randint(0, 4)  # number of 90 degree rotations to perfrom
            if rotations:
                X = np.rot90(X, rotations, axes=(1, 2))
                y = np.rot90(y, rotations, axes=(1, 2))

        # normalize
        if self.normalize_subframes:
            for i in range(X.shape[0]):
                for j in range(X.shape[-1]):
                    X[i, :, :, j] = (X[i, :, :, j] - np.mean(X[i, :, :, j])) / np.std(X[i, :, :, j])

        # rescale
        if self.scaling != (1, 1):
            scale = np.random.uniform(self.scaling[0], self.scaling[1])
            shape_new = [int(X.shape[1]*scale), int(X.shape[2]*scale)]
            shape_new = tuple([16*(d//16) for d in shape_new])  # ensure dimensions are divisible by 16
            X_new = np.zeros((X.shape[0], shape_new[0], shape_new[1], X.shape[3]))
            y_new = np.zeros((y.shape[0], shape_new[0], shape_new[1], y.shape[3]))
            for i in range(self.batch_size):
                X_new[i] = cv2.resize(X[i, :, :], shape_new)
                y_new[i] = cv2.resize(y[i, :, :], shape_new)
            X, y = [X_new.copy(), y_new.copy()]  # todo: is it necessary to copy these variables???

        return X, y

    def __len__(self):
        return self.epoch_size
