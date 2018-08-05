'''
TO DO:
create models for each trained forder by data, and choose model with best val loss automatically
data augmentation
overrepresentation of more challenging frames?
scripts to generate predictions given a video
make interpretable loss function using chris's whisker tracing algorithm
'''

# training settings
dataset_name = 'data\\scaling0.25_filtering50.h5'
use_cpu = True
test_set_portion = .1
lr_init = .001
batch_size = 32
kernel_size = 5
training_epochs = 100
first_layer_filters = 16 # 32 seemed to overfit, 8 seemed to underfit
save_test_predictions = True



# prepare_data settings
whiskers = 4
label_filtering = 50
scaling = 0.25



# make_video settings
model_name = 'filters16_kern5_weights.25-0.000104.hdf5'
vid_name = '180118_KM131.mkv'