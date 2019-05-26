import glob
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_frame(file, percentiles=(0, 100)):
    """
    given name of .tif file, returns image as numpy array normalized from 0-1
    values clipped at percentiles provided by user
    """

    img = Image.open(file)
    img_np = np.array(img, dtype='float32')
    [pix_min, pix_max] = np.percentile(img_np, np.array(percentiles))
    img_np = np.clip((img_np-pix_min) / (pix_max-pix_min), 0, 1)  # scale to between 0 and 1
    return img_np


def preview_vid(folder, frames_to_show=100, fps=30, close_when_done=False, percentiles=(0, 100)):
    """
    opens window and plays movie from sequence of .tif files
    shows the first frames_to_show frames contained in folder
    """

    # initialize window
    im_plot = plt.imshow(np.zeros((1, 1), dtype='float32'), cmap='plasma', vmin=0, vmax=1)
    plt.show()
    files = glob.glob(os.path.join(folder, '*.tif'))
    frames_to_show = min(frames_to_show, len(files))

    for f in tqdm(files[0:frames_to_show]):
        frame = get_frame(f, percentiles)
        im_plot.set_data(frame)
        plt.pause(1/fps)

    if close_when_done:
        plt.close('all')




