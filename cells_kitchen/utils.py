import glob
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from scipy.ndimage import convolve
from scipy.stats import zscore
import matplotlib.pyplot as plt
import json
import cv2



def get_frame(file):
    """
    given name of .tif file, returns image as numpy array normalized from 0-1
    todo: is there a faster way to load a bunch of .tif files?
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

    # normalize image
    imgs = zscore(imgs, axis=0)

    # compute correlation image
    img_corr = convolve(imgs, kernel[np.newaxis, :], mode='constant')
    img_corr = imgs * img_corr
    img_corr = np.mean(img_corr, 0)
    img_corr = scale_img(img_corr)  # scale from 0->1

    return img_corr


def scale_img(img):
    """ scales numpy array between min_val and max_val """
    return (img - np.min(img.flatten())) / np.ptp(img.flatten())


def get_masks(folder, collapse_masks=False, centroid_radius=2, border_thickness=2):
    """
    for folder containing labeled data, returns masks for soma, border of cells, and centroid. returned as 3D bool
    stacks, with one mask per cell, unless collapse_masks is True, in which case max is taken across all cells
    """

    # get image dimensions
    with open(os.path.join(folder, 'info.json')) as f:
        dimensions = json.load(f)['dimensions'][1:3]

    # load labels
    with open(os.path.join(folder, 'regions', 'consensus_regions.json')) as f:
        cell_masks = [np.array(x['coordinates']) for x in json.load(f)]

    # compute masks for each neuron
    masks_soma, masks_border, masks_centroids = \
        [np.zeros((len(cell_masks), dimensions[0], dimensions[1]), dtype=bool) for _ in range(3)]

    for i, cell in enumerate(cell_masks):
        masks_soma[i, cell[:, 0], cell[:, 1]] = True
        _, contour, _ = cv2.findContours(masks_soma[i].astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        masks_border[i] = cv2.drawContours(np.zeros(dimensions), contour, 0, 1, thickness=border_thickness).astype('bool')
        center = np.mean(cell, 0).astype('int')
        masks_centroids[i] = cv2.circle(masks_centroids[i].astype('uint8'), (center[0], center[1]), centroid_radius, 1, thickness=-1)

    # collapse across neurons
    if collapse_masks:
        [masks_soma, masks_border, masks_centroids] = \
            [np.max(x, 0) for x in (masks_soma, masks_border, masks_centroids)]

    return masks_soma, masks_border, masks_centroids


def add_contours(img, contour, color=(1, 0, 0)):
    """given 2D img, and 2D contours, returns 3D image with contours added in color"""

    img_contour = np.repeat(img[:,:,np.newaxis], 3, axis=2)
    inds = np.argwhere(contour)
    img_contour[inds[:, 0], inds[:, 1], :] = np.array(color)

    return img_contour


def enhance_contrast(img, percentiles=(5, 95)):
    """given 2D image, rescales the image between lower and upper percentile limits"""

    limits = np.percentile(img.flatten(), percentiles)
    img = (img-limits[0]) / np.ptp(limits)

    return img












