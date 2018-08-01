from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D
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




def format_img(img, target_dims, apply_min_filter=False, min_filter_kernel=0):
    
    # apply min filter before resizing
    if apply_min_filter:
        img = cv2.erode(img, min_filter_kernel, iterations=1)
    
    # resize image
    img = cv2.resize(img, (target_dims[1], target_dims[0]))
    return img



def show_predictions(X, Y, predictions):
    
    examples_to_show = X.shape[0]
    
    # prepare figure
    plt.close('all')
    fig, axes = plt.subplots(2, examples_to_show, sharex=True, sharey=True)
    channels = Y.shape[-1]
    
    # get rgb values for each whisker
    cmap = plt.cm.spring
    colors = np.zeros((channels,3))
    for channel in range(channels):
        colors[channel,:] = cmap(channel / (channels-1))[0:3]
    
    
    # plot ground true and predictions for each sample
    for col in range(examples_to_show):
        
        # get raw image
        raw_img = X[col][:,:,0]
        raw_img = np.repeat(raw_img[:,:,None], 3, axis=2) # add color dimension
        
        # show ground truth and predictions
        for i, data in enumerate((Y, predictions)):
            colored_labels = np.zeros(((X.shape[1], data.shape[2], 3, channels)), dtype='float32')
            for channel in range(channels):
                colored_labels[:,:,:,channel] =  np.repeat(data[col,:,:,channel,None], 3, axis=2) * colors[channel,:]
            colored_labels = np.amax(colored_labels, axis=3) # collapse across all colors
            merged = np.clip(cv2.addWeighted(colored_labels, 1.0, raw_img, 1.0, 0), 0, 1) # overlay raw image
            axes[i, col].imshow(merged)
            axes[i, col].axis('off')
    plt.tight_layout()







