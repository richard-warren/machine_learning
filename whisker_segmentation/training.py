# WHISKER SEGMENTATION


from utils import create_network, show_predictions, DataGenerator
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import keras.backend as K
from config import test_set_portion, are_labels_binary, data_dir, whiskers, lr_init, first_layer_filters, batch_size, use_cpu, read_h5
from keras.callbacks import EarlyStopping, ModelCheckpoint
import losswise
from losswise.libs import LosswiseKerasCallback
import tables



# set up losswise.com visualization
losswise.set_api_key('9BDAXRBWA')

    
# split into train and test sets
with tables.open_file('%s\\dataset.h5' % (data_dir), 'r') as file:
    total_imgs = file.root.imgs.shape[0]
    img_dims = (file.root.imgs.shape[1], file.root.imgs.shape[2])
all_inds = list(range(0, total_imgs))
np.random.shuffle(all_inds)
train_inds = all_inds[0:int(total_imgs*(1-test_set_portion))]
test_inds = all_inds[int(total_imgs*(1-test_set_portion)):]




# create model and data generators
loss_fcn = 'binary_crossentropy' if are_labels_binary else 'mean_squared_error'
model = create_network((img_dims[0], img_dims[1], 1), whiskers, first_layer_filters, optimizer=Adam(lr=lr_init), loss_fcn=loss_fcn)
if read_h5:
    dataset = tables.open_file('%s\\dataset.h5' % (data_dir), 'r')
params = {'data_dir': data_dir,
          'dataset': dataset if read_h5 else (),
          'img_dims': img_dims,
          'output_channels': whiskers,
          'batch_size': batch_size,
          'shuffle': True,
          'read_h5': read_h5,
          'are_labels_binary': are_labels_binary}
train_generator = DataGenerator(train_inds, **params)
test_generator = DataGenerator(test_inds, **params)



# train, omg!
if use_cpu:
    config = tf.ConfigProto(device_count={'GPU':0})
    sess = tf.Session(config=config)
    K.set_session(sess)
callbacks = [EarlyStopping(patience=10, verbose=1), # stop when validation loss stops increasing
           ModelCheckpoint('models\\weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=10), # save models periodically
           LosswiseKerasCallback(tag='giterdone')] # show progress on losswise.com
model.fit_generator(generator=train_generator, validation_data=test_generator, epochs=100, callbacks=callbacks)



# generate and visualize predictions
X, Y = test_generator[0]
predictions = model.predict(X)
examples_to_show = 3
inds = np.random.choice(range(X.shape[0]), size=examples_to_show, replace=False)
show_predictions(X[inds], Y[inds], predictions[inds])


