"""
TODO:
todo: get sketch of segmentation network going!
visualize activations in network, eg high pass filter section
copy config file to model folder to keep track of settings
express parameters in um rather than pixels using dataset metadata
add option for starting with certain model weights?
add metadata storage to models...
try generating predictions on entire image... // how to change model size but keep weights...
try zscore frames instead of 0-1?
choose labels in training script, not in prepare_training_data
batch norm? // res blocks? // dropout?
dice loss function
"""

# general
data_dir = r'C:\Users\erica and rick\Desktop\cells_kitchen'

# prepare training data
datasets = ['N.04.00.t', 'N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'YST', 'K53', 'J115', 'J123']
border_thickness = 2  # thickness of borders for border labels
summary_frames = 1000  # number of frames per batch when computing summary images
max_batches = 1000  # max number of batches to use for computing summary images