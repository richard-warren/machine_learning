from glob import glob
import cv2
from config import scaling, label_filtering, whiskers, are_labels_binary
import os.path
import shutil
import numpy as np
from tqdm import tqdm



total_imgs = len(list(glob('data\\raw\\frames\\*.png')))


# determine target img demensions
img_dims = cv2.imread('data\\raw\\frames\\img1.png')[:, :, 1].shape # load sample image
img_dims = [int(dim*scaling) for dim in img_dims] # reduce by down_sampling factor
img_dims = [dim-dim%4 for dim in img_dims] # ensure dimensions are divisble by four



# create or overwrite folders for data
data_dir = 'data\\data_%.1fscaling_filtering%i_%s' % (scaling, label_filtering, 'binary' if are_labels_binary else 'gaus')
dirs = [data_dir+'\\labeled', data_dir+'\\frames']
for dir in dirs:
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)



# create kernels for filtering
label_filtering += (label_filtering+1)%2 # ensure this value is odd
min_filter_kernel = np.ones(np.repeat(int(np.ceil(1/scaling)), 2), dtype='uint8') # used to erode image prior to down-sampling to maintain whisker thickness
dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(label_filtering, label_filtering))



# create images and labels
for i in tqdm(range(total_imgs)):
    
    # load and resize raw image
    img = cv2.imread('data\\raw\\frames\\img%i.png' % (i+1))
    img = cv2.erode(img, min_filter_kernel)
    img = cv2.resize(img, (img_dims[1], img_dims[0]))
    cv2.imwrite('%s\\frames\\img%i.png' % (data_dir, i+1), img)
    
    # load labels
    for j in range(whiskers):
        file_name = 'frame%05d_whisker_C%i.png' % (i+1, j)
        
        # create confidence map
        if os.path.isfile('data\\raw\\labeled\\' + file_name):
            label = cv2.imread('data\\raw\\labeled\\' + file_name)
            if are_labels_binary:
                label = cv2.dilate(label, dilation_kernel)
            else:
                label = cv2.GaussianBlur(label.astype('float32'), (label_filtering, label_filtering), 0)
                label = cv2.resize(label, (img_dims[1], img_dims[0]))
            label = label * (255/np.amax(label)).astype('uint8')
        else:
            label = np.zeros(img_dims, dtype='uint8') # set confidence map to all zeros if whisker is not in frame
        
        cv2.imwrite('%s\\labeled\\%s' % (data_dir, file_name), label)