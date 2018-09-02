import cv2
from keras.models import load_model
from config import model_folder, vid_name
import os
from utils import add_labels_to_frame, add_maxima_to_frame
import numpy as np
from tqdm import tqdm
from glob import glob
import scipy
import subprocess



# load model
model = load_model(glob(os.path.join('models', model_folder, '*.hdf5'))[0])
model_dims = model.layers[0].input_shape[2:0:-1] # width X height
multi_output = len(model.outputs)>1
channels = model.output_shape[-1][-1] if multi_output else model.output_shape[-1]

# create vid reader and writer
vid_read = cv2.VideoCapture(os.path.join('videos',vid_name))
input_dims = (int(vid_read.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid_read.get(cv2.CAP_PROP_FRAME_HEIGHT)))
input_fps = vid_read.get(cv2.CAP_PROP_FPS)
total_frames = int(vid_read.get(cv2.CAP_PROP_FRAME_COUNT))

# initialize min filter for down-sampling
min_filter_kernel = np.ones(np.repeat(round(input_dims[0]/model_dims[0]), 2), dtype='uint8')

# create temp folder for images
temp_dir = os.path.join('videos', 'temp')
if os.path.exists(temp_dir):
    for file_name in glob(os.path.join('videos', 'temp', '*.png')):
        os.remove(file_name)
    os.rmdir(temp_dir)
os.makedirs(temp_dir)



for i in tqdm(range(total_frames)):
    got_frame, frame_hires = vid_read.read()
    
    if got_frame:
        
        # get prediction
        frame = cv2.erode(frame_hires, min_filter_kernel)
        frame = cv2.resize(frame, model_dims)[:,:,1]
        frame = frame.astype('float32') / 255
        prediction = model.predict(frame[None,:,:,None])[-1][0,:,:,:] if multi_output else model.predict(frame[None,:,:,None])[0,:,:,:]
        
        # add predicted labels to frame
        frame_hires = frame_hires.astype('float32') / 255
        labeled_frame = add_labels_to_frame(frame_hires[:,:,1], prediction, range(4), whiskers=4)
        
        # write to video
        labeled_frame = np.clip((labeled_frame*255).astype('uint8'), 0, 255)
        labeled_frame = add_maxima_to_frame(labeled_frame, prediction, range(4, channels)) # add maxima to frame
        scipy.misc.toimage(labeled_frame, cmin=0.0, cmax=1.0).save(os.path.join(temp_dir, 'frame%04d.png'%i))
        
        
    else:
        break
vid_read.release()

#vid_write.release()
subprocess.call(['ffmpeg', '-y',
                 '-framerate', str(input_fps),
                 '-i', os.path.join('videos', 'temp', 'frame%04d.png'),
                 '-r', str(input_fps),
                 os.path.join('videos',os.path.splitext(vid_name)[0]+'_labeled.mp4')])

# delete temporary files
for file_name in glob(os.path.join('videos', 'temp', '*.png')):
    os.remove(file_name)
os.rmdir(temp_dir)

