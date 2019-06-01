from config import data_dir, datasets, summary_frames
import numpy as np
import os
from utils import get_frames, get_correlation_image, scale_img, get_masks
from PIL import Image

for d in datasets:

    print('preparing training data for %s' % d)

    # get summary images
    img_stack = get_frames(os.path.join(data_dir, 'datasets', 'images_' + d),
                           frame_num=summary_frames, contiguous=False)
    summaries = {
        'X_corr': get_correlation_image(img_stack),
        'X_mean': scale_img(np.mean(img_stack, 0)),
        'X_max': scale_img(img_stack.max(0)),
        'X_std': scale_img(img_stack.std(0))
    }
    X = np.stack(summaries.values(), axis=2)

    # get targets
    targets = get_masks(os.path.join(data_dir, 'labels', d), collapse_masks=True, centroid_radius=3, border_thickness=2)
    y = np.stack(targets.values(), axis=2)

    # write to disk
    np.savez(os.path.join(data_dir, 'training_data', d), X=X, y=y)

    # write images to disk
    hgt = 600
    for t in (summaries, targets):
        for key, val in t.items():
            file_name = os.path.join(data_dir, 'training_data', d + '_' + key + '.png')
            img = Image.fromarray((val*255).astype('uint8'))
            img = img.resize((int((val.shape[1]/val.shape[0])*hgt), hgt))
            img.save(file_name)

print('all done!')









