# WHISKER SEGMENTATION


from utils import create_network


# settings image size
img_size = (10,10,3)
filters = 32
output_channels = 4



model = create_network(img_size, output_channels, filters)












