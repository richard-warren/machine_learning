## initializations
prefix = "F:\\cells_kitchen_files\\datasets\\images_"
suffixes = ['N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'YST']

## look at some sweet, sweet vids
vid_num = 4

preview_vid(prefix+suffixes[vid_num], frames_to_show=50, fps=100)

## show summary images for vid
vid_num = 4
frames = 500

img_stack = get_frames(prefix+suffixes[vid_num], frames, contiguous=False)

img_corr = get_correlation_image(img_stack)
img_median = np.median(img_stack, 0)
img_max = img_stack.max(0)
img_std = scale_img(img_stack.std(0))

mosaic = np.concatenate((img_corr, img_median, img_max, img_std), 1)
plt.imshow(mosaic)
plt.show()
