from glob import glob
import cv2
from config import scaling, label_filtering, whiskers
import os.path
import numpy as np
import tables
from tqdm import tqdm



total_imgs = len(list(glob('data\\raw\\frames\\*.png')))


# determine target img demensions
img_dims = cv2.imread('data\\raw\\frames\\img1.png')[:, :, 1].shape # load sample image
img_dims = [int(dim*scaling) for dim in img_dims] # reduce by down_sampling factor
img_dims = [dim-dim%4 for dim in img_dims] # ensure dimensions are divisble by four



# create kernels for filtering
label_filtering_odd = label_filtering + (label_filtering+1)%2 # ensure this value is odd
min_filter_kernel = np.ones(np.repeat(int(np.ceil(1/scaling)), 2), dtype='uint8') # used to erode image prior to down-sampling to maintain whisker thickness


file_name = 'data\\scaling%.2f_filtering%i.h5' % (scaling, label_filtering)
with tables.open_file(file_name, 'w') as file: # open h5 file for saving all images and labels
    
    file.create_array(file.root, 'imgs', np.empty((total_imgs, img_dims[0], img_dims[1], 1), dtype='uint8'))
    file.create_array(file.root, 'labels', np.empty((total_imgs, img_dims[0], img_dims[1], whiskers), dtype='float32'))
    
    # create images and labels
    for i in tqdm(range(total_imgs)):
        
        # load and resize raw image
        img = cv2.imread('data\\raw\\frames\\img%i.png' % (i+1))[:,:,1]
        img = cv2.erode(img, min_filter_kernel)
        img = cv2.resize(img, (img_dims[1], img_dims[0]))
        file.root.imgs[i] = img[:,:,None]
        
        # load labels
        for j in range(whiskers):
            file_name = 'frame%05d_whisker_C%i.png' % (i+1, j)
            
            # create confidence map
            if os.path.isfile('data\\raw\\labeled\\' + file_name):
                label = cv2.imread('data\\raw\\labeled\\' + file_name)[:,:,1]
                label = cv2.GaussianBlur(label.astype('float32'), (label_filtering_odd, label_filtering_odd), 0)
                label = cv2.resize(label, (img_dims[1], img_dims[0]))
            else:
                label = np.zeros(img_dims, dtype='float32') # set confidence map to all zeros if whisker is not in frame
            file.root.labels[i,:,:,j] = label