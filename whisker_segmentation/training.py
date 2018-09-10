from utils import DataGenerator
from models import models_dict
from evaluate_model import evaluate_model
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
import losswise
from losswise.libs import LosswiseKerasCallback
import tables
from config import test_set_portion, dataset_name, lr_init, first_layer_filters, batch_size, use_cpu, training_epochs, kernel_size, use_sample_weights, sample_weight_lims, network_structure
from datetime import datetime
import os
import pickle
from glob import glob
from tqdm import tqdm

losswise.set_api_key('9BDAXRBWA') # set up losswise.com visualization



# prepare sample weights
if use_sample_weights:
    angles = np.genfromtxt(os.path.join('data','raw','frame_angles.csv'))[1:,3]
    bin_counts, bin_edges = np.histogram(angles, bins=10)
    frame_bins = np.digitize(angles, bin_edges[:-1])
    
    sample_weights = (1/bin_counts) / np.mean(1/bin_counts) # weights are inverse of bins
    sample_weights = np.clip(sample_weights, sample_weight_lims[0], sample_weight_lims[1]) # make sure the weights aren't too big or small
    sample_weights = sample_weights[frame_bins-1] # give each frame a weight



with tables.open_file(os.path.join('data',dataset_name+'.h5'), 'r') as dataset:
    
    # split into train and test sets
    total_imgs = dataset.root.imgs.shape[0]
    sample_weights = sample_weights[0:total_imgs] if use_sample_weights else []
    all_inds = list(range(0, total_imgs))
    np.random.shuffle(all_inds)
    train_inds = all_inds[0:int(total_imgs*(1-test_set_portion))]
    test_inds = all_inds[int(total_imgs*(1-test_set_portion)):]


    # create model and data generators
    train_generator = DataGenerator(train_inds, dataset, batch_size=batch_size, shuffle=True, sample_weights=sample_weights, 
                                    num_loss_fcns=2 if network_structure=='stacked_hourglass' else 1)
    test_generator = DataGenerator(test_inds, dataset, batch_size=batch_size, shuffle=False, sample_weights=sample_weights, 
                                   num_loss_fcns=2 if network_structure=='stacked_hourglass' else 1)
    

    model = models_dict(network_structure)((train_generator.img_dims[0], train_generator.img_dims[1], 1), train_generator.channels, first_layer_filters, 
                                           kernel_size = kernel_size,
                                           optimizer = Adam(lr=lr_init),
                                           loss_fcn = 'mean_squared_error')

    # train, omg!
    if use_cpu:
        config = tf.ConfigProto(device_count={'GPU':0})
        sess = tf.Session(config=config)
        K.set_session(sess)
         
    model_folder = datetime.now().strftime('%y%m%d_%H.%M.%S')
    model_path = os.path.join('models', model_folder)
    os.makedirs(model_path)
    callbacks = [EarlyStopping(patience=10, verbose=1), # stop when validation loss stops increasing
               ModelCheckpoint(os.path.join(model_path, '%s_filters%i_kern%i_sampleweights%s_weights.{epoch:02d}-{val_loss:.6f}.hdf5'%(model.name, first_layer_filters, kernel_size, 'ON' if use_sample_weights else 'OFF')), save_best_only=True), # save models periodically
               LosswiseKerasCallback(tag='giterdone', display_interval=1)] # show progress on losswise.com
    history = model.fit_generator(generator=train_generator, validation_data=test_generator, epochs=training_epochs, callbacks=callbacks)
    with open(os.path.join(model_path, 'training_history'), 'wb') as training_file:
        pickle.dump(history.history, training_file)
    
    # load best model and delete others
    models_to_delete = glob(os.path.join(model_path,'*hdf5'))[:-1]
    [os.remove(mod) for mod in iter(models_to_delete)]
    model = load_model(glob(os.path.join(model_path, '*.hdf5'))[0])
    
    
        
    # get predictions for test set
    print('generating test set predictions...')
    test_batches = len(test_generator)

    with tables.open_file(os.path.join(model_path,'predictions.h5'), 'w') as file: # open h5 file for saving test images and labels
        file.create_array(file.root, 'imgs',
                          atom = tables.Float32Atom(),
                          shape = (test_batches*batch_size, train_generator.img_dims[0], train_generator.img_dims[1], 1))
        file.create_array(file.root, 'labels',
                          atom = tables.Float32Atom(),
                          shape = (test_batches*batch_size, train_generator.img_dims[0], train_generator.img_dims[1], train_generator.channels))
        file.create_array(file.root, 'predictions',
                          atom = tables.Float32Atom(),
                          shape = (test_batches*batch_size, train_generator.img_dims[0], train_generator.img_dims[1], train_generator.channels))
        file.create_array(file.root, 'test_set_imgs_ids', [ind+1 for ind in test_inds])
        
        # save test set data and predictions
        for i in tqdm(range(test_batches)):
            if network_structure=='stacked_hourglass':
                imgs, _, weights = test_generator[i]
                file.root.imgs[i*batch_size:(i+1)*batch_size] = imgs
                file.root.labels[i*batch_size:(i+1)*batch_size] = _[1]
                _ = model.predict(imgs)
                file.root.predictions[i*batch_size:(i+1)*batch_size] = _[1]
            else:
                imgs, file.root.labels[i*batch_size:(i+1)*batch_size], weights = test_generator[i]
                file.root.imgs[i*batch_size:(i+1)*batch_size] = imgs
                file.root.predictions[i*batch_size:(i+1)*batch_size] = model.predict(imgs)
        
        # copy metadata from training set - is there a way to do this more elegantly?
        file.create_array(file.root, 'whiskers', np.array(dataset.root.whiskers))
        file.create_array(file.root, 'original_dims', np.array(dataset.root.original_dims[test_inds,:]))
        if dataset.root.labels.shape[-1]>dataset.root.whiskers: # if there are more channels than whiskers, it means there are some channels trakcing whisker points
            file.create_array(file.root, 'original_point_coordinates', np.array(dataset.root.original_point_coordinates[test_inds,:,:]))
            file.create_array(file.root, 'downsampled_point_coordinates', np.array(dataset.root.downsampled_point_coordinates[test_inds,:,:]))
            
    
# evaluate model
point_distances = evaluate_model(model_folder, write_imgs=True)
np.save(os.path.join(model_path,'pix_diffs(mean%.3f)' % np.nanmean(point_distances)), point_distances)
print('mean pixel differnce: %.3f' % np.nanmean(point_distances))

