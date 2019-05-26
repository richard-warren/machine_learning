import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

# config stuff
training_data_folder = "local_files\\WEBSITE\\datasets"

# initializations
folders = next(os.walk(os.path.join('.', training_data_folder)))[1]
masks = []
imgs = []

for folder in folders:
    print('collecting data... %s' % folder)

    # load metadata
    with open(os.path.join(training_data_folder, folder, 'info.json')) as f:
        dimensions = json.load(f)['dimensions'][1:3]

    # load images
    img_median = np.array(
        cv2.resize(cv2.imread(os.path.join(training_data_folder, folder, 'projections', 'median_image.png')),
                   tuple(dimensions[::-1])))
    img_corr = np.array(
        cv2.resize(cv2.imread(os.path.join(training_data_folder, folder, 'projections', 'correlation_image.png')),
                   tuple(dimensions[::-1])))
    imgs.append(np.stack((img_median[:, :, 0], img_corr[:, :, 0]), 2))

    # load labels data
    with open(os.path.join(training_data_folder, folder, 'regions', 'consensus_regions.json')) as f:
        neurons = [np.array(x['coordinates']) for x in json.load(f)]

    # turn neurons into single mask
    mask = np.zeros(dimensions, dtype=bool)
    for neuron in neurons:
        mask[neuron[:, 0], neuron[:, 1]] = True
    masks.append(mask)

# plot that shit
fig, axes = plt.subplots(3, 3)
for i in range(len(folders)):
    tiles = np.concatenate((masks[i] * 255, imgs[i][:, :, 0], imgs[i][:, :, 1]), 1)
    axes[np.unravel_index(i, [3, 3])].imshow(tiles)
fig.show()
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()








