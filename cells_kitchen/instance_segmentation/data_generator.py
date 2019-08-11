from cells_kitchen.config import data_dir
from cells_kitchen.instance_segmentation import config as cfg
from keras.utils import Sequence
import numpy as np
import pandas as pd
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
                 fraction_positive_egs=1, jitter=4, negative_eg_distance=8):

        # initialization
        self.datasets = datasets
        self.batch_size = batch_size
        self.subframe_size = subframe_size
        self.epoch_size = epoch_size
        self.rotation = rotation
        self.scaling = scaling
        self.fraction_positive_egs = fraction_positive_egs
        self.jitter = jitter
        self.negative_eg_distance = negative_eg_distance

        # load features and labels into DataFrame
        self.data = pd.DataFrame(index=datasets, columns=['X', 'y', 'negative_eg_mask'])
        for d in datasets:

            data_sub = np.load(os.path.join(data_dir, 'training_data', d + '.npz'), allow_pickle=True)
            X = np.stack([data_sub['X'][()][k] for k in cfg.X_layers], axis=2)
            y = data_sub['neuron_masks']

            # get logical masks representing pixels where pos and neg egs can be centered at
            negative_eg_mask = np.zeros(X.shape[:2], dtype='uint8')
            for n in range(y.shape[0]):  # loop across neurons
                center = np.round(center_of_mass(y[n])).astype('int')
                negative_eg_mask = cv2.circle(
                    negative_eg_mask, (center[1], center[0]), negative_eg_distance, 1, thickness=-1)
            negative_eg_mask = np.invert(negative_eg_mask.astype('bool'))

            self.data.loc[d, :] = (X, y, negative_eg_mask.astype('bool'))

        self.shape_X = (batch_size,) + subframe_size + (self.data.loc[datasets[0], 'X'].shape[-1],)  # batch size X height X width X depth
        self.shape_y = (batch_size,) + subframe_size
        self.dy = np.floor(subframe_size[0]/2).astype('uint8')
        self.dx = np.floor(subframe_size[1]/2).astype('uint8')

    def __getitem__(self, index):

        # gets data for batch
        X = np.zeros(self.shape_X)
        y = np.zeros(self.shape_y)

        for i in range(self.batch_size):
            # select random dataset, eg type, and neuron if positive eg
            dataset_ind = np.random.randint(len(self.datasets))
            is_eg_positive = self.fraction_positive_egs > np.random.uniform(0, 1)

            if is_eg_positive:
                # todo: add jitter // deal with image borders
                neuron_ind = np.random.randint(self.data.y[dataset_ind].shape[0])
                center = np.round(center_of_mass(self.data.y[dataset_ind][neuron_ind])).astype('int')
                x_inds = slice(center[1]-self.dx, center[1]+self.dx)
                y_inds = slice(center[0]-self.dy, center[0]+self.dy)
                X[i] = self.data.X[dataset_ind][y_inds, x_inds]
                y[i] = self.data.y[dataset_ind][neuron_ind, y_inds, x_inds]
            else:
                pass

        # # rotate
        # if self.rotation:
        #     rotations = np.random.randint(4)  # number of 90 degree rotations to perfrom
        #     if rotations:
        #         X = np.rot90(X, rotations, axes=(1, 2))
        #         y = np.rot90(y, rotations, axes=(1, 2))
        #
        # # rescale
        # if self.scaling != (1, 1):
        #     scale = np.random.uniform(self.scaling[0], self.scaling[1])
        #     shape_new = [int(X.shape[1]*scale), int(X.shape[2]*scale)]
        #     shape_new = tuple([16*(d//16) for d in shape_new])  # ensure dimensions are divisible by 16
        #     X_new = np.zeros((X.shape[0], shape_new[0], shape_new[1], X.shape[3]))
        #     y_new = np.zeros((y.shape[0], shape_new[0], shape_new[1], y.shape[3]))
        #     for i in range(self.batch_size):
        #         X_new[i] = cv2.resize(X[i, :, :], shape_new)
        #         y_new[i] = cv2.resize(y[i, :, :], shape_new)
        #     X, y = [X_new.copy(), y_new.copy()]  # todo: is it necessary to copy these variables???

        return X, y

    def __len__(self):
        return self.epoch_size
