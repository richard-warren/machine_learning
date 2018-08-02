# WHISKER SEGMENTATION
'''
TO DO:
data generator
how to make predictions with data generator?
smarter Y normalization
checkpoint saving, training termination rules, and naming models with settings
way to plot loss and training like eddie
'''

from utils import create_network, show_predictions, DataGenerator
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import keras.backend as K
from glob import glob
from config import test_set_portion, are_labels_binary, data_dir, whiskers, lr_init, first_layer_filters, batch_size, use_cpu
import cv2





# split into train and test sets
total_imgs = len(list(glob(data_dir+'\\frames\\*.png')))
all_inds = list(range(0, total_imgs))
np.random.shuffle(all_inds)
train_inds = all_inds[0:int(total_imgs*(1-test_set_portion))]
test_inds = all_inds[int(total_imgs*(1-test_set_portion)):]




# create model and data generators
loss_fcn = 'binary_crossentropy' if are_labels_binary else 'mean_squared_error'
img_dims = cv2.imread(data_dir+'\\frames\\img1.png').shape # determine target image dimensions
model = create_network((img_dims[0], img_dims[1], 1), whiskers, first_layer_filters, optimizer=Adam(lr=lr_init), loss_fcn=loss_fcn)
params = {'data_dir': data_dir,
          'img_dims': img_dims,
          'output_channels': whiskers,
          'batch_size': batch_size,
          'shuffle': True,
          'are_labels_binary': are_labels_binary}
train_generator = DataGenerator(train_inds, **params)
test_generator = DataGenerator(test_inds, **params)



# train, omg!
if use_cpu:
    config = tf.ConfigProto(device_count={'GPU':0})
    sess = tf.Session(config=config)
    K.set_session(sess)
model.fit_generator(generator=train_generator, validation_data=test_generator)



# generate and visualize predictions
X, Y = test_generator.__getitem__(0)
predictions = model.predict(X)
examples_to_show = 6
inds = np.random.choice(range(X.shape[0]), size=examples_to_show, replace=False)
show_predictions(X[inds], Y[inds], predictions)


