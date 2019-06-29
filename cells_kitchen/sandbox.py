## initializations
prefix = "F:\\cells_kitchen_files\\datasets\\images_"
labels_folder = "F:\\cells_kitchen_files\\labels\\"
suffixes = ['N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'YST']

## look at some sweet, sweet vids
vid_num = 0

preview_vid(prefix+'K53', frames_to_show=np.inf, fps=100)

## show summary and target images for vid
vid_num = 2
frames = 500

# get summary images
img_stack = get_frames(prefix+suffixes[vid_num], frames, contiguous=False)
#
img_corr = get_correlation_image(img_stack)
img_mean = scale_img(np.mean(img_stack, 0))
img_max = scale_img(img_stack.max(0))
img_std = scale_img(img_stack.std(0))
summaries = (img_corr, img_mean, img_max, img_std)

# get targets
[masks_soma, masks_border, masks_centroids] = \
    get_masks(labels_folder+suffixes[vid_num], collapse_masks=True, centroid_radius=3)
targets = (masks_soma, masks_border, masks_centroids)

# add label borders
border_color = (.5, 0, 0)
summaries_border = [add_contours(x, masks_border, color=border_color) for x in summaries]

# display those guys
mosaic_summaries = np.concatenate(summaries, 1)
mosaic_targets = np.concatenate(targets, 1)
plt.subplot(2, 1, 1); plt.imshow(mosaic_summaries)
plt.subplot(2, 1, 2); plt.imshow(mosaic_targets)
plt.show()

## load and generate prediction images for model

model_path = r'F:\cells_kitchen_files\models\190629_17.13.59'


from data_generators import DataGenerator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from losswise.libs import LosswiseKerasCallback
import config as cfg
from datetime import datetime
import os
import pickle
from glob import glob

# create model data generators
train_generator = DataGenerator(cfg.train_datasets, batch_size=cfg.batch_size, subframe_size=cfg.subframe_size,
                                normalize_subframes=cfg.normalize_subframes, epoch_size=cfg.epoch_size//cfg.batch_size,
                                rotation=cfg.aug_rotation, scaling=cfg.aug_scaling)
test_generator = DataGenerator(cfg.test_datasets, batch_size=cfg.batch_size, subframe_size=cfg.subframe_size,
                               normalize_subframes=cfg.normalize_subframes, epoch_size=cfg.epoch_size//cfg.batch_size,
                               rotation=cfg.aug_rotation, scaling=cfg.aug_scaling)

model = load_model(glob(os.path.join(model_path, '*.hdf5'))[0])

# get predictions for single batch
X, y = test_generator[0]
y_pred = model.predict(X)
for i in range(X.shape[0]):
    file = os.path.join(model_path, 'prediction%i.png' % i)
    save_prediction_img(file, X[i], y[i], y_pred[i], X_contrast=(0, 100))






