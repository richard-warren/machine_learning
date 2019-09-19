## look at some sweet, sweet vids
import config as cfg
import os
import utils
import numpy as np


vid = r'C:\Users\erica and rick\Desktop\cells_kitchen\neurofinder\train\neurofinder.04.00\images'
preview_vid(vid, frames_to_show=1000, fps=100)

## write sample training images

import utils
# utils.write_sample_imgs()
utils.write_sample_border_imgs(channels=['corr', 'median', 'mean'], contrast=(1,99))


## write sample images with red borders
import utils
import config as cfg
import glob
import os
import numpy as np
from PIL import Image

# settings
channels = ('corr', 'mean', 'median')
height = 800


print('writing sample summary images to disk! omg!')
subfolder = 'neurofinder' if cfg.use_neurofinder else 'caiman'
files = glob.glob(os.path.join(cfg.data_dir, 'training_data', subfolder, '*.npz'))
for f in files:

    # load data
    data = np.load(f)
    X = data['X'][()]
    y = data['y'][()]

    # restrict to requested channels, and borders only for y
    X = dict((k, X[k]) for k in channels)  # restrict to requested channels
    X = np.stack(X.values(), axis=2)
    y = y['borders']  # restrict to requested channels

    # add borders
    img = np.zeros((y.shape[0], y.shape[1]*len(channels), 3))
    for i in range(len(channels)):
        img[:, i*(X.shape[1]):(i+1)*X.shape[1], :] = utils.add_contours(X[:, :, i], y)

    file_name = os.path.join(cfg.data_dir, 'training_data', subfolder, os.path.splitext(f)[0] + '_borders.png')

    img = Image.fromarray((img * 255).astype('uint8'))
    img = img.resize((int((img.width / img.height) * height), height), resample=Image.NEAREST)
    img.save(file_name)