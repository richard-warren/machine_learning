"""
TODO:
todo: new dataset generalization
todo: data aug to get rid of across dataset differences?
add option for starting with certain model weights?
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
datasets = ['N.04.00.t', 'N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'YST', 'K53', 'J115', 'J123']

# training data
border_thickness = 2  # thickness of borders for border labels
summary_frames = 1000  # number of frames to use when computing summary images
max_batches = 1000  # max number of batches to use for computing summary images

# training
X_layers = ['corr', 'median']  # ['corr', 'mean', 'median', 'max', 'std']
y_layers = ['somas', 'centroids']  # ['somas', 'borders', 'centroids']
high_pass_sigma = 15  # std of gaussian based high pass filtering of inptus // set to False to turn off high pass filtering
aug_rotation = True  # whether to apply 0, 90, 180, or 270 degree rotations randomly
aug_scaling = (.75, 1.25)  # min and max image scaling // set to (1, 1) for no scaling
batch_normalization = True
losswise_api_key = '3ZGMSXASM'  # set to False if not using losswise.com
lr_init = .1
normalize_subframes = False  # now this is built into the model when batch_normalization is True
subframe_size = (160, 160)  # each dimension must be divisible by four
test_datasets = ['N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'YST']
train_datasets = ['K53', 'J115', 'J123']
use_cpu = False  # whether to use CPU instead of GPU for training
filters = 32  # seemed to work with as little as 16 // 8 was a little blurrier, which is encouraging...
save_predictions_during_training = True  # set whether to save images of predictions at each epoch end during training

batch_size = 16
epoch_size = 64  # number of image (NOT batches) in an epoch
training_epochs = 5000  # epochs
early_stopping = training_epochs//10  # epochs

