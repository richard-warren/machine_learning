'''
TO DO:
*evenly sample across whisker positions
make interpretable loss function using chris's whisker tracing algorithm
make distance loss function for whisker points
separate trace and point loss functions

create models for each trained forder by data, and choose model with best val loss automatically
data augmentation
scripts to generate predictions given a video
'''

# training settings
dataset_name = 'data\\scaling0.25_points1.h5'
use_cpu = False
test_set_portion = .1
lr_init = .001
batch_size = 32
kernel_size = 5
training_epochs = 100
first_layer_filters = 16 # 32 seemed to overfit, 8 seemed to underfit
save_test_predictions = True
use_sample_weights = True


# prepare_data settings
whiskers = 4
whisker_points = [0] # points along the whiskers to locate
label_filtering = 50 # sigma for whisker trace confidence maps
point_filtering = 5 # sigma for whisker point confidence maps
scaling = 0.25


# make_video settings
model_name = 'filters16_kern5_weights.25-0.000199.hdf5'
vid_name = '180118_KM131.mkv'