## look at some sweet, sweet vids

import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import tifffile


folder = r'C:\Users\erica and rick\Desktop\cells_kitchen\caiman\datasets\images_N.03.00.t'
frames_to_show = 1000
fps = 10
close_when_done = True
frames_to_sample = 100
percentiles = [1, 99.5]

# find images
files = glob.glob(os.path.join(folder, '*.tif*'))
frames_to_show = min(frames_to_show, len(files))

# get pixel percentiles using random sample of images
inds = np.random.choice(frames_to_show, min(frames_to_sample, frames_to_sample))
imgs = tifffile.imread(np.array(files)[inds].tolist()).astype('float32')
lims = np.percentile(imgs, percentiles)

escaping = False


def close_window(event):
    if event.key == 'escape':
        global escaping
        escaping = True
        plt.close()


# initialize window
fig = plt.figure(num=folder, figsize=(6.4, 6.4*(imgs.shape[1]/imgs.shape[2])))
fig.canvas.mpl_connect('key_press_event', close_window)
im = plt.imshow(np.zeros((1, 1), dtype='float32'), cmap='bone', vmin=lims[0], vmax=lims[1])
ax = plt.gca()
ax.set_position([0, 0, 1, 1])
plt.show()

for f in files[0:frames_to_show]:
    frame = tifffile.imread(f).astype('float32')
    im.set_data(frame)
    plt.pause(1/fps)
    if escaping:
        break

if close_when_done:
    plt.close()


## write sample training images

import utils
# utils.write_sample_imgs(X_contrast=(0,99))
# utils.write_sample_border_imgs(channels=['corr', 'median', 'mean'], contrast=(0,99))
utils.write_sample_border_imgs(channels=['corr'], contrast=(0,99))

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

