## initializations
prefix = "F:\\cells_kitchen_files\\datasets\\images_"
labels_folder = "F:\\cells_kitchen_files\\labels\\"
suffixes = ['N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'YST']

## look at some sweet, sweet vids
vid_num = 0

preview_vid(prefix+'K53', frames_to_show=np.inf, fps=100)

## test spatial high pass filtering

file = r'F:\cells_kitchen_files\training_data\N.01.01.npz'
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

data = np.load(file)
img = data['X'][()]['mean']

##
lowpass = ndimage.gaussian_filter(img, 20)
plt.cla()
plt.subplot(2, 2, 1); plt.imshow(img)
plt.subplot(2, 2, 2); plt.imshow(lowpass)
plt.subplot(2, 2, 3); plt.imshow(img-lowpass)

##
from scipy import signal
import numpy as np
a = scipy.signal.gaussian(61, 20, sym=True)
kernel = np.outer(a, a)