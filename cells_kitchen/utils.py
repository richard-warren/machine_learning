import glob
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from scipy.ndimage import convolve
from scipy.stats import zscore
import matplotlib.pyplot as plt


def get_frame(file):
    """
    given name of .tif file, returns image as numpy array normalized from 0-1
    values clipped at percentiles provided by user
    """

    img = np.array(Image.open(file), dtype='float32')
    img = img / np.max(img)
    return img


def get_frames(folder, frame_num=100, contiguous=False):
    """
    given folder name and frames_to_get, gets frames_to_get evenly spaced frames from folder
    """

    frame_num = min(frame_num, 1500)  # todo: this hack prevents user from falling helplessly into a RAMless abyss

    files = glob.glob(os.path.join(folder, '*.tif'))
    frame_num = min(len(files), frame_num)  # ensure requested frames don't exceed frames available
    if contiguous:
        frame_inds = range(frame_num)
    else:
        frame_inds = np.floor(np.linspace(0, len(files)-1, frame_num)).astype('int16')

    img = get_frame(files[0])
    imgs = np.zeros((frame_num, img.shape[0], img.shape[1]))

    for i, f in enumerate(tqdm(frame_inds)):
        imgs[i] = get_frame(files[f])

    return imgs


def preview_vid(folder, frames_to_show=100, fps=30, close_when_done=False):
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
        frame = get_frame(f)
        im_plot.set_data(frame)
        plt.pause(1/fps)

    if close_when_done:
        plt.close('all')


def get_correlation_image(imgs):
    """
    given movie folder, returns image representing temporal correlation between each pixel and surrounding eight pixels
    """

    # define 8 neighbors filter
    kernel = np.ones((3, 3), dtype='float32')
    kernel[1, 1] = 0

    # compute correlation image
    img_corr = convolve(imgs, kernel[np.newaxis, :], mode='constant')
    img_corr = imgs * img_corr
    img_corr = np.mean(img_corr, 0)
    img_corr = scale_img(img_corr)  # scale from 0->1

    return img_corr


def scale_img(img, min_val=0, max_val=1):
    """ scales numpy array between min_val and max_val """
    return (img - np.min(img)) / np.ptp(img.flatten())


