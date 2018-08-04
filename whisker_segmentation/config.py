

# training settings
data_dir= 'data\\0.25scaling_filtering50_gaus'
read_h5 = True
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
are_labels_binary = False
write_imgs = False