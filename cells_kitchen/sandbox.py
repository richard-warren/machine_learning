## look at some sweet, sweet vids
prefix = "F:\\cells_kitchen_files\\datasets\\images_"
labels_folder = "F:\\cells_kitchen_files\\labels\\"
suffixes = ['N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'YST']
vid_num = 0

preview_vid(prefix+'K53', frames_to_show=np.inf, fps=100)

## write sample training images

import utils
utils.write_sample_imgs()