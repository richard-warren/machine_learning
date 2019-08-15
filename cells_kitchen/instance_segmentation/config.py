'''
TODO:
(x) get mask working with all positive egs
(x) get mask and class working with all positive egs
add class prediction to save_prediction_imgs
get mask working with pos and neg egs
get segmentation branch working as well :)
'''

# network
test_datasets = ['N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'YST']
train_datasets = test_datasets.copy()  # ['K53', 'J115', 'J123']
X_layers = ['corr', 'median', 'std']  # summary images to include as input to the network // ['corr', 'mean', 'median', 'max', 'std']
subframe_size = (40, 40)  # cubed root needs to be whole number // 40, 48, 60
filters = 8  # seemed to work with as little as 16 // 8 was a little blurrier, which is encouraging...

# training
mask_weight = 0  #  how much to weight mask vs classification loss during training
fraction_positive_egs = .5  # fraction of training examples with an object in the center
jitter = 2  # object can be jittered +-jitter relative to center
negative_eg_distance = 8  # negative examples must have center at last this far from closest cell center

use_cpu = False  # whether to use CPU instead of GPU for training
aug_rotation = True  # whether to apply 0, 90, 180, or 270 degree rotations randomly
aug_scaling = (.75, 1.25)  # min and max image scaling // set to (1, 1) for no scaling
batch_normalization = True
losswise_api_key = '3ZGMSXASM'  # set to False if not using losswise.com
lr_init = .1
batch_size = 16
epoch_size = 64  # number of image (NOT batches) in an epoch
training_epochs = 5000  # epochs
early_stopping = training_epochs//10  # epochs
save_predictions_during_training = True  # set whether to save images of predictions at each epoch end during training
