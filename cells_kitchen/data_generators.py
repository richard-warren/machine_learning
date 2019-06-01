from keras.utils import Sequence


class DataGenerator(Sequence):
    '''
    each call returns stack of X and ys, whers stack contains batch_size images randomly selected for datasets
    each frame will be square subframe of random size, but no larger than smallest dataset
    locations in frame will also be random
    '''

    def __init__(self, datasets, batch_size=8, ):
        # initialization

        self.datasets = datasets
        self.batch_size = batch_size
        
        self.img_dims = (dataset.root.imgs.shape[1], dataset.root.imgs.shape[2])
        self.channels = dataset.root.labels.shape[-1]

        self.on_epoch_end()  # shuffle inds on initialization

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # gets data for batch

        batch_inds = self.img_inds[index * self.batch_size: (index + 1) * self.batch_size]
        X = self.dataset.root.imgs[batch_inds, :, :, :].astype('float32')
        Y = self.dataset.root.labels[batch_inds, :, :, :].astype('float32')

        # normalize
        for smp in range(len(batch_inds)):
            for channel in range(self.channels):
                if np.max(Y[smp, :, :, channel]) > 0:
                    Y[smp, :, :, channel] = Y[smp, :, :, channel] / np.max(Y[smp, :, :, channel])
        X = X / 255.

        #        print('get batch time: %.2f' % (time.time()-t))
        if self.num_loss_fcns == 1:
            return X, Y, self.sample_weights[batch_inds]
        else:
            return X, [Y for _ in range(self.num_loss_fcns)], [self.sample_weights[batch_inds] for _ in range(
                self.num_loss_fcns)]  # return same Y and smp_weights multiple times if using intermediate supervision

    def on_epoch_end(self):
        # shuffle data at the end of epoch

        if self.shuffle:
            np.random.shuffle(self.img_inds)

