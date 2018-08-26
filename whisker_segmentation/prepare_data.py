from glob import glob
import cv2
from config import scaling, trace_filtering, point_filtering, whiskers, whisker_points, img_limit
import os.path
import numpy as np
import tables
from tqdm import tqdm
#import ipdb







# determine target img demensions
img_dims = cv2.imread('data\\raw\\frames\\img1.png')[:, :, 1].shape # load sample image
img_dims = [int(dim*scaling) for dim in img_dims] # reduce by down_sampling factor
img_dims = [dim-dim%16 for dim in img_dims] # ensure dimensions are divisble by four



# create kernels for filtering
trace_filtering_odd = trace_filtering + (trace_filtering+1)%2 # ensure this value is odd
min_filter_kernel = np.ones(np.repeat(int(np.ceil(1/scaling)), 2), dtype='uint8') # used to erode image prior to down-sampling to maintain whisker thickness


# get whisker point locations
total_imgs = len(list(glob('data\\raw\\frames\\*.png'))) if not img_limit else img_limit
if whisker_points:
    point_locations = np.empty((total_imgs, whiskers, len(whisker_points), 2))
    X, Y = np.meshgrid(range(img_dims[1]), range(img_dims[0]))
    
    for whisker in range(whiskers):
        for point_ind, point in enumerate(iter(whisker_points)):
            point_locations[:, whisker, point_ind, :] = np.genfromtxt('data/raw/C%i_%i.csv' % (whisker, point))[1:total_imgs+1,:] # skip first row in spreadsheet, which contains column headings



file_name = 'data\\scaling%.2f_points%i_tracefiltering_%i_pointfiltering%i_imgs%i.h5' % (scaling, len(whisker_points), trace_filtering, point_filtering, total_imgs)
total_labels = whiskers+ whiskers*len(whisker_points)



with tables.open_file(file_name, 'w') as file: # open h5 file for saving all images and labels
    
    file.create_array(file.root, 'imgs', np.empty((total_imgs, img_dims[0], img_dims[1], 1), dtype='uint8'))
    file.create_array(file.root, 'labels', np.empty((total_imgs, img_dims[0], img_dims[1], total_labels), dtype='float32'))
    file.create_array(file.root, 'whiskers', np.array([whiskers], dtype='uint8'))
    file.create_array(file.root, 'original_dims', np.empty((total_imgs,2), dtype='uint16')) # record dimensions input image before down-sampling
    if whisker_points:
        point_locations_reshaped = np.reshape(point_locations, (total_imgs, whiskers*len(whisker_points),2))
        file.create_array(file.root, 'original_point_coordinates', point_locations_reshaped)
        file.create_array(file.root, 'downsampled_point_coordinates', np.empty(point_locations_reshaped.shape, dtype='float32'))
    
    # create images and labels
    for i in tqdm(range(total_imgs)):
        
        # load image
        img = cv2.imread('data\\raw\\frames\\img%i.png' % (i+1))[:,:,1]
        file.root.original_dims[i,:] = img.shape
        file.root.downsampled_point_coordinates[i] = file.root.original_point_coordinates[i] * (np.divide(img_dims,img.shape))
        
        # resize
        img = cv2.erode(img, min_filter_kernel)
        img = cv2.resize(img, (img_dims[1], img_dims[0]))
        file.root.imgs[i] = img[:,:,None]
        
        # load labels
        for j in range(whiskers):
            img_file_name = 'frame%05d_whisker_C%i.png' % (i+1, j)
            
            # create whisker trace confidence map
            if os.path.isfile('data\\raw\\labeled\\' + img_file_name ):
                label = cv2.imread('data\\raw\\labeled\\' + img_file_name )[:,:,1]
                original_dims = list(label.shape)
                label = cv2.GaussianBlur(label.astype('float32'), (trace_filtering_odd, trace_filtering_odd), 0)
                label = cv2.resize(label, (img_dims[1], img_dims[0]))
            else:
                label = np.zeros(img_dims, dtype='float32') # set confidence map to all zeros if whisker is not in frame
            file.root.labels[i,:,:,j] = label
            
            # create confidence maps for points along whisker
            for k, whisker in enumerate(whisker_points):

                location = point_locations[i,j,k,:]
                if not (location==0).all(): # when both locations are zero it means the whisker has not been tracked
                    location = np.multiply(location, np.divide(img_dims, original_dims))
                    deltas = np.sqrt((np.power(Y-location[0],2) + np.power(X-location[1],2))) # distance of each pixel to whisker point
                    label = np.exp(-deltas / (2*np.power((point_filtering*scaling),2)))
                else:
                    label = np.zeros(img_dims, dtype='float32') # set confidence map to all zeros if whisker is not in frame
                
                file.root.labels[i,:,:,whiskers+j*len(whisker_points)+k] = label

