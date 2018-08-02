from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D
from keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt
import cv2





def create_network(img_size, output_channels, filters=64, optimizer='adam', loss_fcn='mean_squared_error'):
    
    # create fully convolutional network

    x_in = Input(shape=img_size)

    x1 = Conv2D(filters, kernel_size=3, padding="same", activation="relu")(x_in)
    x1 = Conv2D(filters, kernel_size=3, padding="same", activation="relu")(x1)
    x1 = Conv2D(filters, kernel_size=3, padding="same", activation="relu")(x1)
    x1_pool = MaxPooling2D(pool_size=2, strides=2, padding="same")(x1)
    
    x2 = Conv2D(filters*2, kernel_size=3, padding="same", activation="relu")(x1_pool)
    x2 = Conv2D(filters*2, kernel_size=3, padding="same", activation="relu")(x2)
    x2 = Conv2D(filters*2, kernel_size=3, padding="same", activation="relu")(x2)
    x2_pool = MaxPooling2D(pool_size=2, strides=2, padding="same")(x2)
    
    x3 = Conv2D(filters*4, kernel_size=3, padding="same", activation="relu")(x2_pool)
    x3 = Conv2D(filters*4, kernel_size=3, padding="same", activation="relu")(x3)
    x3 = Conv2D(filters*4, kernel_size=3, padding="same", activation="relu")(x3)
    
    x4 = Conv2DTranspose(filters*2, kernel_size=3, strides=2, padding="same", activation="relu", kernel_initializer="glorot_normal")(x3)
    x4 = Conv2D(filters*2, kernel_size=3, padding="same", activation="relu")(x4)
    x4 = Conv2D(filters*2, kernel_size=3, padding="same", activation="relu")(x4)
    
    x_out = Conv2DTranspose(output_channels, kernel_size=3, strides=2, padding="same", activation="linear", kernel_initializer="glorot_normal")(x4)
    
    # compile
    net = Model(inputs=x_in, outputs=x_out, name="whiskerTracer")
    net.compile(optimizer=optimizer, loss=loss_fcn)
    
    # show network summary
    net.summary()
    
    return net







def show_predictions(X, Y, predictions):
    
    examples_to_show = X.shape[0]
    
    # prepare figure
    plt.close('all')
    channels = Y.shape[-1]
    fig, axes = plt.subplots(examples_to_show, channels+1, sharex=True, sharey=True)
    
    
    # get rgb values for each whisker
    cmap = plt.cm.spring
    colors = np.zeros((channels,3), dtype='float32')
    for channel in range(channels):
        colors[channel,:] = cmap(channel / (channels-1))[0:3]
    
    
    # plot ground true and predictions for each sample
    for row in range(examples_to_show):
        
        # get raw image
        raw_img = X[row][:,:,0]
        raw_img = np.repeat(raw_img[:,:,None], 3, axis=2) # add color dimension
        
        # show ground truth
        colored_labels = np.zeros(((X.shape[1], X.shape[2], 3, channels)), dtype='float32')
        for channel in range(channels):
                colored_labels[:,:,:,channel] =  np.repeat(Y[row,:,:,channel,None], 3, axis=2) * colors[channel,:]
        colored_labels = np.amax(colored_labels, axis=3) # collapse across all colors
        merged = np.clip(cv2.addWeighted(colored_labels, 1.0, raw_img, 1.0, 0), 0, 1) # overlay raw image
        axes[row, 0].imshow(merged)
        axes[row, 0].axis('off')    
                
        # show ground truth and predictions
        for channel in range(channels):
            colored_label =  np.repeat(predictions[row,:,:,channel,None], 3, axis=2) * colors[channel,:]
            merged = np.clip(cv2.addWeighted(colored_label, 1.0, raw_img, 1.0, 0), 0, 1) # overlay raw image
            axes[row, channel+1].imshow(merged)
            axes[row, channel+1].axis('off')
    plt.tight_layout()





class DataGenerator(Sequence):
    # keras data generator class
    
    
    def __init__(self, img_inds, data_dir, img_dims, output_channels=1, batch_size=16, shuffle=True, are_labels_binary=False):
        # initialization
        
        self.data_dir = data_dir
        self.img_inds = img_inds
        self.img_dims = img_dims
        self.output_channels = output_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.are_labels_binary = are_labels_binary
        self.on_epoch_end() # shuffle inds on initialization
        
        
    def __len__(self):
        # number of batches per epoch
        
        return int(np.floor(len(self.img_inds) / self.batch_size))
    
    
    def __getitem__(self, index):
        # gets data for batch
        
        # initialize containers
        batch_inds = self.img_inds[index*self.batch_size : (index+1)*self.batch_size]
        X = np.empty((self.batch_size, self.img_dims[0], self.img_dims[1], 1), dtype='float32')
        Y = np.empty((self.batch_size, self.img_dims[0], self.img_dims[1], self.output_channels),
                     dtype='bool' if self.are_labels_binary else 'float32')
        
        
        # get data
        for i, batch_ind in enumerate(batch_inds):
            
            # load image
            img = cv2.imread('%s\\frames\\img%i.png' % (self.data_dir, batch_ind+1))[:, :, 1].astype('float32') / 255
            X[i] = img[:,:,None]
            
            # load labels
            for j in range(self.output_channels):
                file = "%s\\labeled\\frame%05d_whisker_C%i.png" % (self.data_dir, batch_ind+1, j)
                label = cv2.imread(file)[:, :, 1]
                Y[i,:,:,j] = label.astype('float32') / 255
    
        # normalize Y values
        if self.are_labels_binary:
            Y = Y>0
        else:
            Y = Y / np.max(Y)

        return X, Y
    

    def on_epoch_end(self):
        # shuffle data at the end of epoch
        
        if self.shuffle == True:
            np.random.shuffle(self.img_inds)










