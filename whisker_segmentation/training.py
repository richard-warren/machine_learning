# WHISKER SEGMENTATION
'''
to do:
use all images, and split training, test sets
pick up training where left off
checkpoint saving, training termination rules
data generator?
try with resnet50 on top
'''

from utils import create_network
from keras.optimizers import Adam
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os.path
from keras.applications.resnet50 import ResNet50




resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

# settings
binary_labels = True
lr_init = .001
loss_fcn = 'binary_crossentropy' if binary_labels else 'mean_squared_error'
batch_size = 32
total_samples = 5000
filters = 16
output_channels = 3
data_dir = 'data\\frames'
labels_dir = 'data\\labeled'
img_dims = (548,640) # currently must be divisible by 4
dilation = 40
gauss_blurring = 75 # needs to be odd number


# load and augment data
X = np.empty((total_samples, img_dims[0], img_dims[1], 1), dtype='float32')
Y = np.empty((total_samples, img_dims[0], img_dims[1], output_channels),
             dtype='bool' if binary_labels else 'float32')
dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilation,dilation)) # used to thicken the whiskers up when using binary labels

for i in range(total_samples):
    X[i] = cv2.imread('%s\\img%i.png' % (data_dir, i+1))[0:img_dims[0], 0:img_dims[1], 1, None].astype('float32') / 255
    print(i/total_samples)
    for j in range(output_channels):
        if os.path.isfile("%s\\frame%05d_whisker_C%i.png" % (labels_dir, i+1, j+1)):
            img = cv2.imread("%s\\frame%05d_whisker_C%i.png" % (labels_dir, i+1, j+1))[0:img_dims[0], 0:img_dims[1], 1]
            if binary_labels:
                img = cv2.dilate(img, dilation_kernel)
            else:
                img = cv2.GaussianBlur(img.astype('float32'), (gauss_blurring,gauss_blurring),0)
        else:
            img = np.zeros((img_dims[0], img_dims[1])) # set confidence map to all zeros if whisker is not in frame
        Y[i,:,:,j] = img


# normalize Y values
if not binary_labels:
    Y = Y / np.max(Y)


model = create_network(X[0].shape, output_channels, filters, optimizer=Adam(lr=lr_init), loss_fcn=loss_fcn)
model.fit(X, Y, batch_size=batch_size, epochs=10, verbose=1)






# show prediction
predictions = model.predict(X[1:10,])











