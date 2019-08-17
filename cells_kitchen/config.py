'''
TODO:
fix breaking when only one output
check neg eg segmentation network generator
fix training data - why discrepancy btwn summary imgs and labels?
max finding algo
put everything together!
visualize activations in network, eg high pass filter section
copy config file to model folder to keep track of settings
express parameters in um rather than pixels using dataset metadata
dice loss function
'''

# general
data_dir = r'C:\Users\erica and rick\Desktop\cells_kitchen'

# prepare training data
datasets = ['N.04.00.t', 'N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'YST', 'K53', 'J115', 'J123']
border_thickness = 2  # (pixels) thickness of borders for border labels
centroid_radius = 3  # radius of circles at center of neuron mask in target
summary_frames = 1000  # number of frames per batch when computing summary images (1000)
max_batches = 1000  # max number of batches to use for computing summary images (1000)
