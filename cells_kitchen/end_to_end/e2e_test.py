from cells_kitchen.region_proposal.config import X_layers as rp_channels
from cells_kitchen.instance_segmentation.config import X_layers as is_channels
from cells_kitchen import utils
import numpy as np
from keras.models import load_model
import skimage.measure
import skimage.feature
from tqdm import tqdm
import matplotlib.pyplot as plt


data_name = r'C:\Users\erica and rick\Desktop\cells_kitchen\training_data\N.00.00.npz'
rp_model_name = r'C:\Users\erica and rick\Desktop\cells_kitchen\models\region_proposal\191006_21.09.28\unet.92-0.355555.hdf5'
is_model_name = r'C:\Users\erica and rick\Desktop\cells_kitchen\models\instance_segmentation\191008_18.56.30\segnet.221-0.275403.hdf5'

# load data and models
data = np.load(data_name, allow_pickle=True)['X'][()]
data_rp = np.stack([data[k] for k in rp_channels], axis=-1)
data_is = np.stack([data[k] for k in is_channels], axis=-1)
model_rp = load_model(rp_model_name)
model_is = load_model(is_model_name)

# get region proposals
rp = model_rp.predict(np.expand_dims(data_rp, 0)).squeeze()

# get local maxima
maxima = skimage.feature.peak_local_max(rp, min_distance=2, threshold_abs=.2, indices=False)
maxima = skimage.measure.label(maxima, 8)
maxima = skimage.measure.regionprops(maxima)
centroids = np.array([m.centroid for m in maxima])

# perform instance segmentation at each maximum
subframe_size = model_is.input_shape[1:3]
subframes, segmentations, scores = [], [], []

for m in tqdm(maxima):
    position = (int(m.centroid[0]-subframe_size[0]/2),
                int(m.centroid[1]-subframe_size[1]/2),
                subframe_size[0],
                subframe_size[1])
    subframe = utils.get_subimg(data_is, position)
    segmentation, score = model_is.predict(subframe[None,:,:,:])
    segmentations.append(segmentation.squeeze())
    scores.append(score[0][0])
    subframes.append(utils.get_subimg(rp, position).squeeze())

mean_cell = np.mean(np.array(segmentations), 0)
mean_cell = mean_cell / mean_cell.max()


## show image

cell = 20

if 'fig' not in locals():
    plt.close('all')
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
fig.show()

# show centroids
ax[0,0].clear()
ax[0,0].imshow(rp, cmap='gray')
# ax[0,0].plot(centroids[cell, 1], centroids[cell, 0], 'r.')
ax[0,0].scatter(centroids[:,1], centroids[:,0], 3, c=scores, cmap=plt.get_cmap('rainbow'))

ax[1,0].clear()
ax[1,0].imshow(data_rp[:,:,0], cmap='gray')

ax[1,1].clear()
ax[1,1].imshow(data_rp[:,:,1], cmap='gray')

## add cells in different colors
score_thresh = .2
cmap = plt.get_cmap('gist_rainbow')
bg = np.zeros((rp.shape[0]+2*subframe_size[0], rp.shape[1]+2*subframe_size[1], 3))
img = bg.copy()

cell_maps = []

for i, s in tqdm(enumerate(segmentations)):
    if scores[i] > score_thresh:
        r, c = int(centroids[i][0] - subframe_size[0]/2), int(centroids[i][1] - subframe_size[1]/2)
        cell_map = bg.copy()
        s_temp = s
        s_temp = np.repeat(s_temp[:, :, None], 3, 2)
        cell_map[r+subframe_size[0]:r+subframe_size[0]*2, c+subframe_size[1]:c+subframe_size[1]*2] = s_temp
        color = cmap(np.random.rand())[:-1]
        cell_maps.append(cell_map * color)

img = np.array(cell_maps).max(0)
img = img[subframe_size[0]:subframe_size[0]+rp.shape[0], subframe_size[1]:subframe_size[1]+rp.shape[1]]
ax[0,1].clear()
ax[0,1].imshow(img, vmin=0, vmax=1)
