# WHISKER SEGMENTATION


from utils import create_network, show_predictions, DataGenerator
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
import keras.backend as K
from config import test_set_portion, dataset_name, lr_init, first_layer_filters, batch_size, use_cpu, training_epochs
from keras.callbacks import EarlyStopping, ModelCheckpoint
import losswise
from losswise.libs import LosswiseKerasCallback
import tables
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
                           optimizer = Adam(lr=lr_init),
                           loss_fcn = 'mean_squared_error')
    


    # train, omg!
    if use_cpu:
        config = tf.ConfigProto(device_count={'GPU':0})
        sess = tf.Session(config=config)
        K.set_session(sess)
    callbacks = [EarlyStopping(patience=10, verbose=1), # stop when validation loss stops increasing
               ModelCheckpoint('models\\weights.{epoch:02d}-{val_loss:.6f}.hdf5', period=5), # save models periodically
               LosswiseKerasCallback(tag='giterdone')] # show progress on losswise.com
    model.fit_generator(generator=train_generator, validation_data=test_generator, epochs=training_epochs, callbacks=callbacks)
    
    
    
    # generate test set predictions
    test_batches = len(test_generator)
    X_test = np.empty((test_batches*batch_size, train_generator.img_dims[0], train_generator.img_dims[1], 1), dtype='float32')
    Y_test = np.empty((test_batches*batch_size, train_generator.img_dims[0], train_generator.img_dims[1], train_generator.channels), dtype='float32')
    for i in range(test_batches):
        inds = np.arange((i)*batch_size, (i+1)*batch_size)
        X_test[inds], Y_test[inds] = test_generator[i]
    predictions_test = model.predict_generator(test_generator)
    
    


# show predictions
show_predictions(X_test, Y_test, predictions_test, examples_to_show=3)

