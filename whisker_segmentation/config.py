'''
TO DO:
    -add prediction saving
    -write to videos!
'''

# training settings
dataset_name = 'data\\scaling0.25_filtering50.h5'
use_cpu = False
test_set_portion = .05
lr_init = .001
batch_size = 32
training_epochs = 100
first_layer_filters = 16



# prepare_data settings
whiskers = 4
label_filtering = 50
scaling = 0.25