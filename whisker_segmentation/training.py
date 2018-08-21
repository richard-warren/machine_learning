from utils import create_network, show_predictions, DataGenerator
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
import losswise
from losswise.libs import LosswiseKerasCallback
import tables
from config import test_set_portion, dataset_name, lr_init, first_layer_filters, batch_size, use_cpu, training_epochs, save_test_predictions, kernel_size, use_sample_weights
from datetime import datetime
import os

losswise.set_api_key('9BDAXRBWA') # set up losswise.com visualization

    


with tables.open_file(dataset_name, 'r') as dataset:
    
    # split into train and test sets
    total_imgs = dataset.root.imgs.shape[0]
    all_inds = list(range(0, total_imgs))
    np.random.shuffle(all_inds)
    train_inds = all_inds[0:int(total_imgs*(1-test_set_portion))]
    test_inds = all_inds[int(total_imgs*(1-test_set_portion)):]


    # create model and data generators
    train_generator = DataGenerator(train_inds, dataset, batch_size=batch_size, shuffle=True)
    test_generator = DataGenerator(test_inds, dataset, batch_size=batch_size, shuffle=False)
    model = create_network((train_generator.img_dims[0], train_generator.img_dims[1], 1), train_generator.channels, first_layer_filters, 
                           kernel_size = kernel_size,
                           optimizer = Adam(lr=lr_init),
                           loss_fcn = 'mean_squared_error')
    
    # !!! prepare sample weights
    
    

    # train, omg!
    if use_cpu:
        config = tf.ConfigProto(device_count={'GPU':0})
        sess = tf.Session(config=config)
        K.set_session(sess)
    model_folder = os.path.join('models', datetime.now().strftime('%y%m%d_%H.%M.%S'))
    os.makedirs(model_folder)
    callbacks = [EarlyStopping(patience=5, verbose=1), # stop when validation loss stops increasing
               ModelCheckpoint(os.path.join(model_folder, 'filters%i_kern%i_weights.{epoch:02d}-{val_loss:.6f}.hdf5'%(first_layer_filters, kernel_size)), save_best_only=True), # save models periodically
               LosswiseKerasCallback(tag='giterdone')] # show progress on losswise.com
    model.fit_generator(generator=train_generator, validation_data=test_generator, epochs=training_epochs, callbacks=callbacks)
    
    # load best model and remove all others
    models_to_delete = [os.path.join(model_folder,f) for f in os.listdir(model_folder)[:-1]]
    [os.remove(mod) for mod in iter(models_to_delete)]
    model = load_model(os.path.join(model_folder, os.listdir(model_folder)[0]))
    
    # generate test set predictions
    if save_test_predictions:
        
        # get X, Y, and predictions for test set
        test_batches = len(test_generator)
        X_test = np.empty((test_batches*batch_size, train_generator.img_dims[0], train_generator.img_dims[1], 1), dtype='float32')
        Y_test = np.empty((test_batches*batch_size, train_generator.img_dims[0], train_generator.img_dims[1], train_generator.channels), dtype='float32')
        for i in range(test_batches):
            inds = np.arange((i)*batch_size, (i+1)*batch_size)
            X_test[inds], Y_test[inds] = test_generator[i]
        predictions_test = model.predict_generator(test_generator)
        
        # save to h5 file
        with tables.open_file(os.path.join(model_folder,'predictions.h5'), 'w') as file: # open h5 file for saving test images and labels
            file.create_array(file.root, 'imgs', X_test)
            file.create_array(file.root, 'labels', Y_test)
            file.create_array(file.root, 'predictions', predictions_test)
            file.create_array(file.root, 'test_set_imgs_ids', [ind+1 for ind in test_inds])
    
    


# show predictions
show_predictions(X_test, Y_test, predictions_test, examples_to_show=3)

