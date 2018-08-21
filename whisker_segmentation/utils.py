from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D
from keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt
import cv2



def create_network(img_size, output_channels, filters=64, kernel_size=3, optimizer='adam', loss_fcn='mean_squared_error'):
    
    # create fully convolutional network

    x_in = Input(shape=img_size)

    x1 = Conv2D(filters, kernel_size=kernel_size, padding="same", activation="relu")(x_in)
    x1 = Conv2D(filters, kernel_size=kernel_size, padding="same", activation="relu")(x1)
    x1 = Conv2D(filters, kernel_size=kernel_size, padding="same", activation="relu")(x1)
    x1_pool = MaxPooling2D(pool_size=2, strides=2, padding="same")(x1)
    
    x2 = Conv2D(filters*2, kernel_size=kernel_size, padding="same", activation="relu")(x1_pool)
    x2 = Conv2D(filters*2, kernel_size=kernel_size, padding="same", activation="relu")(x2)
    x2 = Conv2D(filters*2, kernel_size=kernel_size, padding="same", activation="relu")(x2)
    x2_pool = MaxPooling2D(pool_size=2, strides=2, padding="same")(x2)
    
    x3 = Conv2D(filters*4, kernel_size=kernel_size, padding="same", activation="relu")(x2_pool)
    x3 = Conv2D(filters*4, kernel_size=kernel_size, padding="same", activation="relu")(x3)
    x3 = Conv2D(filters*4, kernel_size=kernel_size, padding="same", activation="relu")(x3)
    
    x4 = Conv2DTranspose(filters*2, kernel_size=3, strides=2, padding="same", activation="relu", kernel_initializer="glorot_normal")(x3)
    x4 = Conv2D(filters*2, kernel_size=3, padding="same", activation="relu")(x4)
    x4 = Conv2D(filters*2, kernel_size=3, padding="same", activation="relu")(x4)
    
    x_out = Conv2DTranspose(output_channels, kernel_size=3, strides=2, padding="same", activation="linear", kernel_initializer="glorot_normal")(x4)
    
    # compile
    net = Model(inputs=x_in, outputs=x_out, name="whisker_tracer")
    net.compile(optimizer=optimizer, loss=loss_fcn)
    
    # show network summary
    net.summary()
    
    return net






def create_network_deep(img_size, output_channels, filters=64, kernel_size=3, optimizer='adam', loss_fcn='mean_squared_error'):
    
    # create fully convolutional network

    x_in = Input(shape=img_size)

    x1 = Conv2D(filters, kernel_size=kernel_size, padding="same", activation="relu")(x_in)
    x1 = Conv2D(filters, kernel_size=kernel_size, padding="same", activation="relu")(x1)
    x1 = Conv2D(filters, kernel_size=kernel_size, padding="same", activation="relu")(x1)
    x1_pool = MaxPooling2D(pool_size=2, strides=2, padding="same")(x1)
    
    x2 = Conv2D(filters*2, kernel_size=kernel_size, padding="same", activation="relu")(x1_pool)
    x2 = Conv2D(filters*2, kernel_size=kernel_size, padding="same", activation="relu")(x2)
    x2 = Conv2D(filters*2, kernel_size=kernel_size, padding="same", activation="relu")(x2)
    x2_pool = MaxPooling2D(pool_size=2, strides=2, padding="same")(x2)
    
    x3 = Conv2D(filters*4, kernel_size=kernel_size, padding="same", activation="relu")(x2_pool)
    x3 = Conv2D(filters*4, kernel_size=kernel_size, padding="same", activation="relu")(x3)
    x3 = Conv2D(filters*4, kernel_size=kernel_size, padding="same", activation="relu")(x3)
    x3_pool = MaxPooling2D(pool_size=2, strides=2, padding="same")(x3)
    
    x4 = Conv2D(filters*8, kernel_size=kernel_size, padding="same", activation="relu")(x3_pool)
    x4 = Conv2D(filters*8, kernel_size=kernel_size, padding="same", activation="relu")(x4)
    x4 = Conv2D(filters*8, kernel_size=kernel_size, padding="same", activation="relu")(x4)
    
    x5 = Conv2DTranspose(filters*2, kernel_size=3, strides=2, padding="same", activation="relu", kernel_initializer="glorot_normal")(x4)
    x5 = Conv2D(filters*4, kernel_size=3, padding="same", activation="relu")(x5)
    x5 = Conv2D(filters*4, kernel_size=3, padding="same", activation="relu")(x5)
    
    x6 = Conv2DTranspose(filters*2, kernel_size=3, strides=2, padding="same", activation="relu", kernel_initializer="glorot_normal")(x5)
    x6 = Conv2D(filters*2, kernel_size=3, padding="same", activation="relu")(x6)
    x6 = Conv2D(filters*2, kernel_size=3, padding="same", activation="relu")(x6)
    
    x_out = Conv2DTranspose(output_channels, kernel_size=3, strides=2, padding="same", activation="linear", kernel_initializer="glorot_normal")(x6)
    
    # compile
    net = Model(inputs=x_in, outputs=x_out, name="whisker_tracer")
    net.compile(optimizer=optimizer, loss=loss_fcn)
    
    # show network summary
    net.summary()
    
    return net





def show_predictions(X, Y, predictions, examples_to_show=3):
    
    
    # prepare figure
    channels = Y.shape[-1]
    fig, axes = plt.subplots(examples_to_show, channels+1, sharex=True, sharey=True)
    inds = np.random.choice(range(X.shape[0]), size=examples_to_show, replace=False)
    
    
    # plot ground true and predictions for each sample
    for row in range(examples_to_show):
        
        # get raw image
        raw_img = X[inds[row],:,:,0]
        
        # show ground truth
        labeled_img = add_labels_to_frame(raw_img, Y[inds[row]], range(channels))
        axes[row, 0].imshow(labeled_img)
        axes[row, 0].axis('off')    
                
        # show ground truth and predictions
        for channel in range(channels):
            labeled_img = add_labels_to_frame(raw_img, predictions[inds[row]], iter([channel]))
            axes[row, channel+1].imshow(labeled_img)
            axes[row, channel+1].axis('off')
    plt.tight_layout()





class DataGenerator(Sequence):
    # keras data generator class
    
    
    def __init__(self, img_inds, dataset, batch_size=16, shuffle=True):
        # initialization
        
        self.img_inds = img_inds
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.img_dims = (dataset.root.imgs.shape[1], dataset.root.imgs.shape[2])
        self.channels = dataset.root.labels.shape[-1]
        
        self.on_epoch_end() # shuffle inds on initialization
        
        
    def __len__(self):
        # number of batches per epoch
        
        return int(np.floor(len(self.img_inds) / self.batch_size))
    
    
    def __getitem__(self, index):
        # gets data for batch
        
        # get data from h5 file
        batch_inds = self.img_inds[index*self.batch_size : (index+1)*self.batch_size]
        X = self.dataset.root.imgs[batch_inds,:,:,:].astype('float32')
        Y = self.dataset.root.labels[batch_inds,:,:,:].astype('float32')
    
        # normalize
        # should replace with a one liner
#        if np.max(Y)>0:
#            Y = Y / np.max(Y)
        for smp in range(len(batch_inds)):
            for channel in range(self.channels):
                if np.max(Y[smp,:,:,channel])>0:
                    Y[smp,:,:,channel] = Y[smp,:,:,channel] / np.max(Y[smp,:,:,channel])            
        X = X / 255

        return X, Y
    

    def on_epoch_end(self):
        # shuffle data at the end of epoch
        
        if self.shuffle:
            np.random.shuffle(self.img_inds)



def add_labels_to_frame(frame, labels, channels_to_show, add_maxima=False):
    
    
    # get rgb values for each whisker
    channels = labels.shape[-1]
    cmap = plt.cm.jet
    colors = np.zeros((channels,3), dtype='float32')
    color_inds = np.linspace(0,1,channels) # not really inds, but evenly spaced values between zero and one
    for channel in range(channels):
        colors[channel,:] = cmap(color_inds[channel])[0:3]
        
    # get colored labels
    colored_labels = np.empty(((labels.shape[0], labels.shape[1], 3, channels)), dtype='float32')
    for channel in channels_to_show:
            colored_labels[:,:,:,channel] =  np.repeat(labels[:,:,channel,None], 3, axis=2) * colors[channel,:]
    colored_labels = np.mean(colored_labels, axis=3) # collapse across all colors
    if np.max(colored_labels)>0:
        colored_labels = colored_labels / np.max(colored_labels)
    
    # upsample labels if necessary
    if not frame.shape==labels.shape:
        colored_labels = cv2.resize(colored_labels, (frame.shape[1], frame.shape[0]))
        
    # merge frame with colored labels
    frame = np.repeat(frame[:,:,None], 3, axis=2) # add color dimension to frame
    merged = np.clip(cv2.addWeighted(colored_labels, 1.0, frame, 1.0, 0), 0, 1) # overlay raw image
    
    return merged
        






