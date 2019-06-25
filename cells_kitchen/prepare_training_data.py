import glob
import config as cfg
import numpy as np
import os
from utils import get_frames, get_correlation_image, scale_img, get_masks
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
    _ = get_frames(folder, frame_inds=range(1), progress_bar=False)
    batches = min(total_frames // cfg.summary_frames, cfg.max_batches)
    all_summaries = np.zeros((batches, _.shape[1], _.shape[2], 4))

    for b in tqdm(range(all_summaries.shape[0])):

        # print('batch frames:: %i-%i' % (batch_inds[b], batch_inds[b]+cfg.summary_frames))

        img_stack = get_frames(folder, frame_inds=np.arange(batch_inds[b], batch_inds[b]+cfg.summary_frames), progress_bar=False)
        summaries_batch = {
            'X_corr': get_correlation_image(img_stack),
            'X_mean': np.mean(img_stack, 0),
            'X_max': img_stack.max(0),
            'X_std': img_stack.std(0)
        }
        all_summaries[b] = np.stack(summaries_batch.values(), axis=2)

    # collapse across summary images and scale from 0-1
    X = all_summaries.max(0)
    for x in range(X.shape[-1]):
        X[x] = scale_img(X[x])

    summaries = {
        'X_corr': X[:,:,0],
        'X_mean': X[:,:,1],
        'X_max': X[:,:,2],
        'X_std': X[:,:,3]
    }

    # get targets
    targets = get_masks(os.path.join(cfg.data_dir, 'labels', d),
                        collapse_masks=True, centroid_radius=3, border_thickness=cfg.border_thickness)
    targets = {k: targets[k] for k in cfg.y_layers}
    y = np.stack(targets.values(), axis=2)

    # write to disk
    np.savez(os.path.join(cfg.data_dir, 'training_data', d), X=X, y=y)

    # write images to disk
    hgt = 600
    for t in (summaries, targets):
        for key, val in t.items():
            file_name = os.path.join(cfg.data_dir, 'training_data', d + '_' + key + '_new.png')
            img = Image.fromarray((val*255).astype('uint8'))
            img = img.resize((int((val.shape[1]/val.shape[0])*hgt), hgt))
            img.save(file_name)

print('all done!')









