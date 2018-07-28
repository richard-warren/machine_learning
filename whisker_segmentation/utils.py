from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import Sequence




def create_network(img_size, output_channels, filters=64):
    
    # define fully convolutional network

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
    net = Model(inputs=x_in, outputs=x_out, name="LeapCNN")
    net.compile(optimizer=Adam(), loss="mean_squared_error")
    
    # show network summary
    net.summary()







class DataGenerator(Sequence):
    
    def __init__(self, list_IDs, labels, batch_size=8, dim=(32,32,32), n_channels=1, n_classes=10, shuffle=True):
        
    