import glob
import config as cfg
import numpy as np
import os
from utils import get_frames, get_correlation_image, scale_img, get_masks, save_prediction_img
from PIL import Image
from tqdm import tqdm
import ipdb as ipdb

for d in cfg.datasets:

    print('preparing training data for %s' % d)

    # get summary images
    folder = os.path.join(cfg.data_dir, 'datasets', 'images_' + d)
    total_frames = len(glob.glob(os.path.join(folder, '*.tif')))
    batch_inds = np.arange(0, total_frames, cfg.summary_frames)

    # initialize image stack
    _ = get_frames(folder, frame_inds=0)
    batches = min(total_frames // cfg.summary_frames, cfg.max_batches)
    all_summaries = np.zeros((batches, _.shape[0], _.shape[1], 4))
    X = dict.fromkeys(['corr', 'mean', 'median', 'max', 'std'], np.zeros((2, 4, 6)))

    for b in tqdm(range(all_summaries.shape[0])):
        img_stack = get_frames(folder, frame_inds=np.arange(batch_inds[b], batch_inds[b]+cfg.summary_frames))
        all_summaries[b, :, :, 0] = get_correlation_image(img_stack)
        all_summaries[b, :, :, 1] = np.mean(img_stack, 0)
        all_summaries[b, :, :, 2] = img_stack.max(0)
        all_summaries[b, :, :, 3] = img_stack.std(0)

    # collapse across summary images and scale from 0-1
    X = all_summaries.max(0)
    X =
    for x in range(X.shape[-1]):
        X[:, :, x] = scale_img(X[:, :, x])

    # get targets
    targets = get_masks(os.path.join(cfg.data_dir, 'labels', d),
                        collapse_masks=True, centroid_radius=3, border_thickness=cfg.border_thickness)
    targets = {k: targets[k] for k in cfg.y_layers}
    y = np.stack(targets.values(), axis=2)

    # write to disk
    np.savez(os.path.join(cfg.data_dir, 'training_data', d), X=X, y=y)

    # write images to disk
    file_name = os.path.join(cfg.data_dir, 'training_data', d + '.png')
    save_prediction_img(file_name, X, y, X_contrast=(0, 100))


print('all done!')









