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






def show_predictions(X, Y, predictions):
    
    examples = X.shape[0]
    
    # generate predictions
    example_inds = np.random.choice(range(X.shape[0]), size=examples, replace=False)
    
    # prepare figure
    plt.close('all')
    fig, axes = plt.subplots(2, examples, sharex=True, sharey=True)
    channels = Y.shape[-1]
    
    # prepare frames with solid colors
    cmap = plt.cm.spring
    color_frames = np.zeros((X.shape[1], X.shape[2], 3, channels), dtype='float32')
    for channel in range(channels):
        rgb = cmap(channel / (channels-1))[0:3]
        for color in range(3):
            color_frames[:,:,color,channel] = rgb[color]
    
    
    for col in range(examples):
        
        # get raw image
        ind = example_inds[col]
        raw_img = X[ind][:,:,0]
        raw_img = np.repeat(raw_img[:,:,None], 3, axis=2) # add color dimension
        
        # show ground truth
        colored_labels = np.zeros(color_frames.shape, dtype='float32')
        for channel in range(channels):
            temp = Y[ind,:,:,channel]
            temp= np.repeat(temp[:,:,None], 3, axis=2)
            colored_labels[:,:,:,channel] =  temp * color_frames[:,:,:,channel]
        colored_labels = np.amax(colored_labels, axis=3)
        merged = cv2.addWeighted(colored_labels, 1.0, raw_img, 1.0, 0)
        merged = np.clip(merged,0,1)
        axes[0, col].imshow(merged)
        
        # show prediction
        colored_labels = np.zeros(color_frames.shape, dtype='float32')
        for channel in range(channels):
            temp = predictions[col,:,:,channel]
            temp= np.repeat(temp[:,:,None], 3, axis=2)
            colored_labels[:,:,:,channel] =  temp * color_frames[:,:,:,channel]
        colored_labels = np.amax(colored_labels, axis=3)
        merged = cv2.addWeighted(colored_labels, 1.0, raw_img, 1.0, 0)
        merged = np.clip(merged,0,1)
        axes[1, col].imshow(merged)
        
        # pimp axis
        for row in range(2):
            axes[row,col].axis('off')
    
    plt.tight_layout()
