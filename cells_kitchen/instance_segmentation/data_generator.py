from keras.utils import Sequence
import numpy as np
import pandas as pd
from cells_kitchen.instance_segmentation import config
import os
import cv2
import ipdb
from scipy.ndimage.measurements import center_of_mass
import cv2


class DataGenerator(Sequence):
    '''
    each call returns stack of X and ys, whers stack contains batch_size images randomly selected for datasets
    each frame will be square subframe of random size, but no larger than smallest dataset
    locations in frame will also be random
    '''

    def __init__(self, datasets, batch_size=8, subframe_size=(40, 40), epoch_size=64, rotation=True, scaling=(1, 1),
                 fraction_positive_egs=.8, jitter=4, negative_eg_distance=8):

        # initialization
        self.datasets = datasets
        self.batch_size = batch_size
        self.subframe_size = subframe_size
        self.epoch_size = epoch_size
        self.rotation = rotation
        self.scaling = scaling

        # load features and labels into DataFrame
        self.data = pd.DataFrame(index=datasets, columns=['X', 'y', 'centroid_mask'])  # centroid mask is
        for d in datasets:

            data_sub = np.load(os.path.join(cfg.data_dir, 'training_data', d + '.npz'), allow_pickle=True)
            X = np.stack([data_sub['X'][()][k] for k in cfg.X_layers], axis=2)
            y = data_sub['neuron_masks']

            # get centroid mask, which is used to determine which pixels are negative_eg_distance from another neuron's
            # center of mass // mask is logical matrix with ones everywhere except circles centered at each neuron's
            # center of mass
            centroid_mask = np.zeros(X.shape[:2], dtype='uint8')
            for n in range(y.shape[0]):  # loop across neurons
                center = np.round(center_of_mass(y[n])).astype('int')
                centroid_mask = cv2.circle(centroid_mask, (center[1], center[0]), negative_eg_distance, 1, thickness=-1)
            centroid_mask = np.invert(centroid_mask.astype('bool'))

            self.data.loc[d, :] = (X, y, centroid_mask)

        self.shape_X = (batch_size,) + subframe_size + (self.data.loc[datasets[0], 'X'].shape[-1],)  # batch size X height X width X depth
        self.shape_y = (batch_size,) + subframe_size

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
