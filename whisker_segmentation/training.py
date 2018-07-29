# WHISKER SEGMENTATION
'''
to do - 
*test with training, test set
max instead of downsampling
predict specific whiskers AND all whiskers
checkpoint saving, training termination rules, and naming models with settings
data generator
way to plot loss and training like eddie
'''

from utils import create_network, show_predictions
from keras.optimizers import Adam
import numpy as np
import cv2
import os.path
import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import train_test_split



# settings
test_size = .05
use_cpu=True
binary_labels = False
lr_init = .001
loss_fcn = 'binary_crossentropy' if binary_labels else 'mean_squared_error'
batch_size = 32
total_samples = 1000
training_epochs = 100
filters = 32
output_channels = 3
data_dir = 'data\\frames'
labels_dir = 'data\\labeled'
img_dims = (548,640)
down_sampling = True
dilation = 40
gauss_blurring = 50



# initializations
if down_sampling:
    img_dims = [int(dim/2) for dim in img_dims]
    dilation, gauss_blurring = int(dilation/2), int(gauss_blurring/2)
    if dilation%2==0:
        dilation += 1
    if gauss_blurring%2==0:
        gauss_blurring += 1
img_dims = [dim-dim%4 for dim in img_dims] # ensure dimensions are divisble by 4    

X = np.empty((total_samples, img_dims[0], img_dims[1], 1), dtype='float32')
Y = np.empty((total_samples, img_dims[0], img_dims[1], output_channels),
             dtype='bool' if binary_labels else 'float32')

dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilation,dilation)) # used to thicken the whiskers up when using binary labels



# load and augment data
for i in range(total_samples):
    
    # load and resize raw image
    img = cv2.imread('%s\\img%i.png' % (data_dir, i+1))[:, :, 1].astype('float32') / 255
    img = cv2.resize(img, (img_dims[1], img_dims[0]))
    img = img[:,:,None]
    X[i]=img
    
    # load labels
    for j in range(output_channels):
        if os.path.isfile("%s\\frame%05d_whisker_C%i.png" % (labels_dir, i+1, j+1)):
            
            # load and resize confidence map
            img = cv2.imread("%s\\frame%05d_whisker_C%i.png" % (labels_dir, i+1, j+1))[:, :, 1]
            img = cv2.resize(img, (img_dims[1], img_dims[0]))
            
            # modify confidence map
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


# split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)



# compile model
model = create_network(X[0].shape, output_channels, filters, optimizer=Adam(lr=lr_init), loss_fcn=loss_fcn)


# train!
if use_cpu:
    config = tf.ConfigProto(device_count={'GPU':0})
    sess = tf.Session(config=config)
    K.set_session(sess)
model.fit(X_train, Y_train, batch_size=batch_size, epochs=training_epochs, verbose=1, shuffle=True)



# generate and show predictions
examples_to_show = 6
inds = np.random.choice(range(X_test.shape[0]), size=examples_to_show, replace=False)
predictions = model.predict(X_test[inds])
show_predictions(X_test[inds], Y_test[inds], predictions)



