from data_generators import DataGenerator
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
import losswise
from losswise.libs import LosswiseKerasCallback
import config as cfg
from datetime import datetime
import os
import pickle
from glob import glob
import models
import utils
import numpy as np

if cfg.losswise_api_key:
    losswise.set_api_key(cfg.losswise_api_key)  # set up losswise.com visualization

# create model data generators
train_generator = DataGenerator(cfg.train_datasets, batch_size=cfg.batch_size, subframe_size=cfg.subframe_size,
                                normalize_subframes=cfg.normalize_subframes, epoch_size=cfg.epoch_size//cfg.batch_size)
test_generator = DataGenerator(cfg.test_datasets, batch_size=cfg.batch_size, subframe_size=cfg.subframe_size,
                               normalize_subframes=cfg.normalize_subframes, epoch_size=cfg.epoch_size//cfg.batch_size)

# create model
model = models.unet(train_generator.shape_X[1:], train_generator.shape_y[-1], filters=cfg.filters)
# model = models.unet((None, None, train_generator.shape_X[-1]), train_generator.shape_y[-1], filters=cfg.filters)

# train, omg!
if cfg.use_cpu:
    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config)
    K.set_session(sess)

model_folder = datetime.now().strftime('%y%m%d_%H.%M.%S')
model_path = os.path.join(cfg.data_dir, 'models', model_folder)
os.makedirs(model_path)
callbacks = [EarlyStopping(patience=cfg.early_stopping, verbose=1), # stop when validation loss stops increasing
             ModelCheckpoint(os.path.join(model_path, '%s.{epoch:02d}-{val_loss:.6f}.hdf5' % model.name), save_best_only=True)]
if cfg.losswise_api_key:
    callbacks.append(LosswiseKerasCallback(tag='giterdone', display_interval=1))
history = model.fit_generator(generator=train_generator, validation_data=test_generator,
                              epochs=cfg.training_epochs, callbacks=callbacks)

with open(os.path.join(model_path, 'training_history'), 'wb') as training_file:
    pickle.dump(history.history, training_file)

# load best model and delete others
models_to_delete = glob(os.path.join(model_path, '*hdf5'))[:-1]
[os.remove(mod) for mod in iter(models_to_delete)]
model = load_model(glob(os.path.join(model_path, '*.hdf5'))[0])

# get predictions for single batch
# import importlib
# importlib.reload(utils)

X, y = test_generator[0]
y_pred = model.predict(X)
for i in range(X.shape[0]):
    utils.save_prediction_img(X[i], y[i], y_pred[i],
                              os.path.join(model_path, 'prediction%i.png' % i), X_contrast=(0, 100))
