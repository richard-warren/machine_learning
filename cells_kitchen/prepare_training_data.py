from cells_kitchen import config as cfg
from cells_kitchen import utils
import glob
import numpy as np
import os
from tqdm import tqdm
import multiprocessing as mp


def prep_training_data(d):
    
    print('preparing training data for %s...' % d)

    # get summary images
    if cfg.use_neurofinder:
        folder = os.path.join(cfg.data_dir, 'neurofinder', 'train', 'neurofinder.' + d, 'images')
    else:
        folder = os.path.join(cfg.data_dir, 'caiman', 'datasets', 'images_' + d)
    total_frames = len(glob.glob(os.path.join(folder, '*.tif*')))
    summary_frames = min(cfg.summary_frames,
                         total_frames)  # make sure we don't look for more frames than exist in the dataset
    batch_inds = np.arange(0, total_frames, summary_frames)

    # initialize image stack
    img0 = utils.get_frames(folder, frame_inds=0)
    batches = min(total_frames // summary_frames, cfg.max_batches)
    summary_titles = ['corr', 'mean', 'median', 'max', 'std']
    X = {key: np.zeros((batches, img0.shape[0], img0.shape[1])) for key in summary_titles}

    # get summary images for each batch in video
    for b in (range(batches) if cfg.parallelize else tqdm(range(batches))):
        img_stack = utils.get_frames(folder, frame_inds=np.arange(batch_inds[b], batch_inds[b] + summary_frames))

        X['corr'][b] = utils.get_correlation_image(img_stack)
        X['mean'][b] = np.mean(img_stack, 0)
        X['median'][b] = np.median(img_stack, 0)
        X['max'][b] = img_stack.max(0)
        X['std'][b] = img_stack.std(0)

    # collapse across summary images and scale from 0-1
    X['corr'] = utils.scale_img(X['corr'].max(0))
    X['mean'] = utils.scale_img(X['max'].max(0))
    X['median'] = utils.scale_img(X['median'].max(0))
    X['max'] = utils.scale_img(X['max'].max(0))
    X['std'] = utils.scale_img(X['std'].mean(0))

    # get targets
    if cfg.use_neurofinder:
        label_folder = os.path.join(cfg.data_dir, 'neurofinder', 'train', 'neurofinder.' + d)
    else:
        label_folder = os.path.join(cfg.data_dir, 'caiman', 'labels', d)
    y = utils.get_targets(label_folder, collapse_masks=True,
                          centroid_radius=cfg.centroid_radius, border_thickness=cfg.border_thickness)

    # get tensor of masks for each individual neuron (used by segmentation network only)
    neuron_masks = utils.get_targets(label_folder, collapse_masks=False)
    neuron_masks = neuron_masks['somas']  # keep only the soma masks

    # store data for model training
    subfolder = 'neurofinder' if cfg.use_neurofinder else 'caiman'
    training_data_folder = os.path.join(cfg.data_dir, 'training_data', subfolder)
    if not os.path.exists(training_data_folder):
        os.makedirs(training_data_folder)
    np.savez(os.path.join(training_data_folder, d), X=X, y=y, neuron_masks=neuron_masks)

    print('done preparing training data for %s...' % d)


if __name__ == '__main__':
    if cfg.parallelize:
        print('parallelizing training data preparation using %i cores' % cfg.cores)
        pool = mp.Pool(cfg.cores)
        pool.map_async(prep_training_data, cfg.datasets)
        pool.close()
        pool.join()
    else:
        [prep_training_data(d) for d in cfg.datasets]

    # write sample images to disk
    utils.write_sample_imgs(X_contrast=(5, 99))
    print('all done!')


