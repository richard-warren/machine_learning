"""
TODO:
todo: add datasets!
todo: image augmentation (rotation, scaling)
summary images over whole videos // and MAX correlation image
try with holdout test set
add metadata storage to models...
try generating predictions on entire image... // how to change model size but keep weights...
try zscore frames instead of 0-1?
choose labels in training script, not in prepare_training_data
batch norm? // res blocks? // dropout?
dice loss function
"""

# general
data_dir = "F:\\cells_kitchen_files\\"
datasets = ['N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'YST']

# training data
y_layers = ['somas', 'centroids']  # ['somas', 'borders', 'centroids']
border_thickness = 2

# training
losswise_api_key = '3ZGMSXASM'  # set to False if not using losswise.com
lr_init = .001
normalize_subframes = False
subframe_size = (160, 160)  # each dimension must be divisible by four
test_datasets = datasets.copy()
train_datasets = datasets.copy()
summary_frames = 1000  # number of frames to use when computing summary images
use_cpu = False  # whether to use CPU instead of GPU for training
filters = 64

batch_size = 16
epoch_size = 64  # number of image (NOT batches) in an epoch
training_epochs = 1000  # epochs
early_stopping = training_epochs//10  # epochs

