'''
TO DO:
switch to n-dimensional upsampling
should load frames and make predictions in batches, would be faster
'''

import cv2
from keras.models import load_model
from config import model_name, vid_name
import os
from utils import add_labels_to_frame
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interpn


# load model
model = load_model('models\\'+model_name)
model_dims = model.layers[0].input_shape[2:0:-1] # width X height
channels = model.output_shape[-1]

# create vid reader and writer
vid_read = cv2.VideoCapture('videos\\'+vid_name)
input_dims = (int(vid_read.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_read.get(cv2.CAP_PROP_FRAME_HEIGHT)))
input_fps = vid_read.get(cv2.CAP_PROP_FPS)
total_frames = int(vid_read.get(cv2.CAP_PROP_FRAME_COUNT))

file_name, _ = os.path.splitext(vid_name)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
vid_write = cv2.VideoWriter('videos\\'+file_name+'_edited.avi', fourcc, input_fps, input_dims)

# initialize min filter for down-sampling
min_filter_kernel = np.ones(np.repeat(round(input_dims[0]/model_dims[0]), 2), dtype='uint8')

# interpolation parameters
#x, y, z = [np.linspace(1,dim,dim) for dim in (model_dims[1], model_dims[0], channels)]
#xi, yi, zi = np.linspace(1,model_dims[1],input_dims[1]), np.linspace(1,model_dims[0],input_dims[0]), z


for i in tqdm(range(total_frames)):
    got_frame, frame_hires = vid_read.read()
    
    if got_frame:
        
        # min filter (to preserve whiskers) and resize
        frame = cv2.erode(frame_hires, min_filter_kernel)
        frame = cv2.resize(frame, model_dims)[:,:,1]
        frame = frame.astype('float32') / 255
        
        # get frame prediction and upsampled to input vid size
        prediction = model.predict(frame[None,:,:,None])[0,:,:,:]
        prediction_upsampled = np.empty((input_dims[1], input_dims[0], model.output_shape[-1]))
        for channel in range(channels):
            prediction_upsampled[:,:,channel] = cv2.resize(prediction[:,:,channel], input_dims)
        frame_hires = frame_hires.astype('float32') / 255
        labeled_frame = add_labels_to_frame(frame_hires[:,:,1], prediction_upsampled)
        
        # write to video
        labeled_frame = np.clip((labeled_frame*255).astype('uint8'), 0, 255)
        vid_write.write(labeled_frame)
        
    else:
        break

vid_read.release()
vid_write.release()