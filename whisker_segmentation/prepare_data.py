from glob import glob
import cv2
from config import scaling, label_filtering, point_filtering, whiskers, whisker_points
import os.path
import numpy as np
import tables
from tqdm import tqdm
#import ipdb



total_imgs = len(list(glob('data\\raw\\frames\\*.png')))


# determine target img demensions
img_dims = cv2.imread('data\\raw\\frames\\img1.png')[:, :, 1].shape # load sample image
img_dims = [int(dim*scaling) for dim in img_dims] # reduce by down_sampling factor
img_dims = [dim-dim%4 for dim in img_dims] # ensure dimensions are divisble by four



# create kernels for filtering
label_filtering_odd = label_filtering + (label_filtering+1)%2 # ensure this value is odd
min_filter_kernel = np.ones(np.repeat(int(np.ceil(1/scaling)), 2), dtype='uint8') # used to erode image prior to down-sampling to maintain whisker thickness


# get whisker point locations
if whisker_points:
    point_locations = np.empty((total_imgs, whiskers, len(whisker_points), 2))
    X, Y = np.meshgrid(range(img_dims[1]), range(img_dims[0]))
    
    for whisker in range(whiskers):
        for point_ind, point in enumerate(iter(whisker_points)):
            point_locations[:, whisker, point_ind, :] = np.genfromtxt('data/raw/C%i_%i.csv' % (whisker, point))[1:,:] # skip first row in spreadsheet, which contains column headings
            


file_name = 'data\\scaling%.2f_points%i.h5' % (scaling, len(whisker_points))
total_labels = whiskers+ whiskers*len(whisker_points)
#total_imgs = 1000 # uncomment to troubleshoot


with tables.open_file(file_name, 'w') as file: # open h5 file for saving all images and labels
    
    file.create_array(file.root, 'imgs', np.empty((total_imgs, img_dims[0], img_dims[1], 1), dtype='uint8'))
    file.create_array(file.root, 'labels', np.empty((total_imgs, img_dims[0], img_dims[1], total_labels), dtype='float32'))
    file.create_array(file.root, 'labels_trace_inds', list(range(0, (whiskers)*(len(whisker_points)+1), len(whisker_points)+1)))
    
    # create images and labels
    for i in tqdm(range(total_imgs)):
        
        # load and resize raw image
        img = cv2.imread('data\\raw\\frames\\img%i.png' % (i+1))[:,:,1]
        img = cv2.erode(img, min_filter_kernel)
        img = cv2.resize(img, (img_dims[1], img_dims[0]))
        file.root.imgs[i] = img[:,:,None]
        
        # load labels
        label_ind = 0
        for j in range(whiskers):
            file_name = 'frame%05d_whisker_C%i.png' % (i+1, j)
            
            # create whisker trace confidence map
            if os.path.isfile('data\\raw\\labeled\\' + file_name):
                label = cv2.imread('data\\raw\\labeled\\' + file_name)[:,:,1]
                original_dims = list(label.shape)
                label = cv2.GaussianBlur(label.astype('float32'), (label_filtering_odd, label_filtering_odd), 0)
                label = cv2.resize(label, (img_dims[1], img_dims[0]))
            else:
                label = np.zeros(img_dims, dtype='float32') # set confidence map to all zeros if whisker is not in frame
            file.root.labels[i,:,:,label_ind] = label
            label_ind += 1
            
            # create confidence maps for points along whisker
            for k, whisker in enumerate(whisker_points):

                location = point_locations[i,j,k,:]
                if not (location==0).all(): # when both locations are zero it means the whisker has not been tracked
                    location = np.multiply(location, np.divide(img_dims, original_dims))
                    deltas = np.sqrt((np.power(Y-location[0],2) + np.power(X-location[1],2))) # distance of each pixel to whisker point
                    label = np.exp(-deltas / (2*point_filtering^2))
                else:
                    label = np.zeros(img_dims, dtype='float32') # set confidence map to all zeros if whisker is not in frame
                
                file.root.labels[i,:,:,label_ind] = label
                label_ind += 1
                
            






