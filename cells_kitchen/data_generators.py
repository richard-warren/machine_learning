from keras.utils import Sequence
import numpy as np
import pandas as pd
from config import data_dir
import os
import ipdb as ipdb


class DataGenerator(Sequence):
    '''
    each call returns stack of X and ys, whers stack contains batch_size images randomly selected for datasets
    each frame will be square subframe of random size, but no larger than smallest dataset
    locations in frame will also be random
    todo: subframe size randomize // random rotation // random scaling
    '''

    def __init__(self, datasets, batch_size=8, subframe_size=(100, 100)):
        # initialization

        self.datasets = datasets
        self.batch_size = batch_size
        self.subframe_size = subframe_size

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
            # ipdb.set_trace()
            X[i] = self.data.loc[self.datasets[b], 'X'][x_inds, y_inds]
            y[i] = self.data.loc[self.datasets[b], 'y'][x_inds, y_inds]

        return X, y

    def __len__(self):
        return 1
