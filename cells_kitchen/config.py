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
parallelize = False  # whether to use parallel processing when creating summary images
cores = 4  # how many CPU cores to use for parallel processes

# prepare training data
# datasets = ['N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t',
#             'YST', 'K53', 'J115', 'J123',
#             'nf.01.00', 'nf.02.01', 'nf.04.01']
datasets = ['N.00.00', 'nf.01.00', 'nf.02.01', 'nf.04.01']
border_thickness = 2  # (pixels) thickness of borders for border labels
centroid_radius = 2  # radius of circles at center of neuron mask in target
summary_frames = 500  # number of frames per batch when computing summary images (1000)
max_batches = 1000  # max number of batches to use for computing summary images (1000)
