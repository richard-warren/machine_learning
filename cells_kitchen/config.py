'''
TODO:
check neg eg segmentation network generator
max finding algo
put everything together!
visualize activations in network, eg high pass filter section
copy config file to model folder to keep track of settings
express parameters in um rather than pixels using dataset metadata
dice loss function
'''

# general
data_dir = r'C:\Users\erica and rick\Desktop\cells_kitchen'
use_neurofinder = False  # whether to use neurofinder data // if False, caiman dataset is used
parallelize = True  # whether to create training sets in parallel
cores = 4  # how many CPU cores to use for parallel processes

# prepare training data
datasets = ['N.04.00.t', 'N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'YST', 'K53', 'J115', 'J123']  # caiman
# datasets = ['00.00', '00.01', '00.02', '00.03', '00.04', '00.05', '00.06', '00.07', '00.08', '00.09', '00.10', '00.11', '01.00', '01.01', '02.00', '02.01', '03.00', '04.00', '04.01']  # neurofinder
border_thickness = 1  # (pixels) thickness of borders for border labels
centroid_radius = 3  # radius of circles at center of neuron mask in target
summary_frames = 1000  # number of frames per batch when computing summary images (1000)
max_batches = 1000  # max number of batches to use for computing summary images (1000)
