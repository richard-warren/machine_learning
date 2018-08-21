'''
TO DO:
*model evaluation scripts (generate images for everything)
evenly sample across whisker positions
make interpretable loss function using chris's whisker tracing algorithm
make distance loss function for whisker points
separate trace and point loss functions

data augmentation
scripts to generate predictions given a video
'''

# training settings
dataset_name = 'data\\scaling0.25_points1_tracefiltering_50_pointfiltering20.h5'
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
whisker_points = [0,7] # points along the whiskers to locate
trace_filtering = 50 # sigma for whisker trace confidence maps
point_filtering = 20 # sigma for whisker point confidence maps
scaling = 0.25


# make_video and evaluate model settings
model_folder = '180820_18.55.11'
vid_name = '180118_KM131.mkv'