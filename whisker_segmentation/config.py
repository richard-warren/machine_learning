'''
TO DO:
*retrain on entire dataset
add maxima to prediction images
look into and try different network architectures (leap alternatives, understand location refinement in dlc)
evenly sample across whisker positions
make interpretable loss function using chris's whisker tracing algorithm
make distance loss function for whisker points
separate trace and point loss functions

data augmentation
scripts to generate predictions given a video
'''

# training settings
dataset_name = 'scaling0.25_points2_tracefiltering_25_pointfiltering5'
use_cpu = False
test_set_portion = .1
lr_init = .001
batch_size = 32
kernel_size = 5
training_epochs = 100
first_layer_filters = 16 # 32 seemed to overfit, 8 seemed to underfit
use_sample_weights = True


# prepare_data settings
whiskers = 4
whisker_points = [0,7] # points along the whiskers to locate
trace_filtering = 25 # sigma for whisker trace confidence maps
point_filtering = 5 # sigma for whisker point confidence maps
scaling = 0.25


# make_video and evaluate model settings
model_folder = '180820_18.55.11'
vid_name = '180118_KM131.mkv'