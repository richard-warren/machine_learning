# WHISKER SEGMENTATION
'''
TO DO:
*data generator
smarter Y normalization
checkpoint saving, training termination rules, and naming models with settings
way to plot loss and training like eddie
'''

from utils import create_network, show_predictions, format_img
from keras.optimizers import Adam
import numpy as np
import cv2
import os.path
import tensorflow as tf
import keras.backend as K
from sklearn.model_selection import train_test_split



# settings
test_size = .05
use_cpu = False
binary_labels = False
lr_init = .001
loss_fcn = 'binary_crossentropy' if binary_labels else 'mean_squared_error'
batch_size = 16
total_samples = 2000
training_epochs = 1000
filters = 16
output_channels = 4
data_dir = 'data\\frames'
labels_dir = 'data\\labeled'
img_dims = (548,640)
down_sampling = 0.5
dilation = 40
gauss_blurring = 50




# initializations
img_dims = [int(dim*down_sampling) for dim in img_dims]
img_dims = [dim-dim%4 for dim in img_dims] # ensure dimensions are divisble by 4    

dilation, gauss_blurring = int(dilation*down_sampling), int(gauss_blurring*down_sampling)
dilation, gauss_blurring = [x + (x+1)%2 for x in (dilation, gauss_blurring)] # dilation and gauss_blurring are odd

X = np.empty((total_samples, img_dims[0], img_dims[1], 1), dtype='float32')
Y = np.empty((total_samples, img_dims[0], img_dims[1], output_channels),
             dtype='bool' if binary_labels else 'float32')

dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilation,dilation)) # used to thicken the whiskers up when using binary labels
min_filter_kernel = np.ones((int(np.ceil(1/down_sampling)), int(np.ceil(1/down_sampling))), dtype='uint8') # used to erode image prior to down-sampling to maintain whisker thickness




# load and augment data
print('loading data...')
for i in range(total_samples):
    
    # load and resize raw image
    img = cv2.imread('%s\\img%i.png' % (data_dir, i+1))[:, :, 1].astype('float32') / 255
    img = format_img(img, img_dims, apply_min_filter=True, min_filter_kernel=min_filter_kernel)
    img = img[:,:,None]
    X[i]=img
    
    # load labels
    for j in range(output_channels):
        if os.path.isfile("%s\\frame%05d_whisker_C%i.png" % (labels_dir, i+1, j)):
            
            # load create confidence map
            label = cv2.imread("%s\\frame%05d_whisker_C%i.png" % (labels_dir, i+1, j))[:, :, 1]
            if binary_labels:
                label = cv2.dilate(label, dilation_kernel)
            else:
                label = cv2.GaussianBlur(label.astype('float32'), (gauss_blurring,gauss_blurring),0)
            label = format_img(label, img_dims, apply_min_filter=False) # resize
            
        else:
            label = np.zeros(img_dims) # set confidence map to all zeros if whisker is not in frame
        Y[i,:,:,j] = label

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



# generate and visualize predictions
examples_to_show = 6
inds = np.random.choice(range(X_test.shape[0]), size=examples_to_show, replace=False)
predictions = model.predict(X_test[inds])
show_predictions(X_test[inds], Y_test[inds], predictions)




