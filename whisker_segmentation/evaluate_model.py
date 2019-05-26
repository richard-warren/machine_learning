import os
from utils import add_labels_to_frame
import numpy as np
import tables
from tqdm import tqdm
import scipy.misc







def evaluate_model(model_to_evaluate, write_imgs=True):
    
    # create folder for image
    if write_imgs:
        results_dir = os.path.join('models', model_to_evaluate, 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    
    with tables.open_file(os.path.join('models', model_to_evaluate, 'predictions.h5'), 'r') as file:
        
        img_num = file.root.predictions.shape[0]
        img_dims = file.root.predictions.shape[1:3]
        channels = file.root.predictions.shape[-1]
        whiskers = file.root.whiskers[0]
        point_distances = np.full((img_num, channels-whiskers), np.nan)
        
        for i in tqdm(range(file.root.predictions.shape[0])):
            
            raw_img = file.root.imgs[i,:,:,0]
            labels = file.root.labels[i,:,:,:]
            predictions = file.root.predictions[i,:,:,:]
            label_locations = file.root.downsampled_point_coordinates[i]
            
            for ind, chan in enumerate(range(whiskers, channels)):
                x_pred, y_pred = np.unravel_index(np.argmax(predictions[:,:,chan]), predictions.shape[0:2])
                x_true, y_true = label_locations[ind]
                if not (x_true==0 and y_true==0):
                    point_distances[i,ind] = np.sqrt(np.power(x_pred-x_true,2) + np.power(y_pred-y_true,2))
            
            
            # write image to disk
            if write_imgs:
                tiles = np.empty((img_dims[0]*2, img_dims[1]*channels, 3))
                for j in range(channels):        
                    
                    # ground truth
                    tiles[0:img_dims[0], j*img_dims[1]:(j+1)*img_dims[1], :] =\
                        add_labels_to_frame(raw_img, labels, iter([j]), data_max=1)
                    
                    # predictions
                    tiles[img_dims[0]:, j*img_dims[1]:(j+1)*img_dims[1], :] = \
                        add_labels_to_frame(raw_img, predictions, iter([j]), data_max=1)
                    
                # save image
                scipy.misc.toimage(tiles, cmin=0.0, cmax=1.0).save(os.path.join(results_dir, 'frame%i.png' % file.root.test_set_imgs_ids[i]))
                
    return point_distances
