## initializations
prefix = "F:\\cells_kitchen_files\\datasets\\images_"
labels_folder = "F:\\cells_kitchen_files\\labels\\"
suffixes = ['N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'YST']

## look at some sweet, sweet vids
vid_num = 0

preview_vid(prefix+suffixes[vid_num], frames_to_show=50, fps=100)

## show summary and target images for vid
vid_num = 0
frames = 1000

# get summary images
img_stack = get_frames(prefix+suffixes[vid_num], frames, contiguous=False)
##
img_corr = get_correlation_image(img_stack)
img_mean = scale_img(np.mean(img_stack, 0))
img_max = scale_img(img_stack.max(0))
img_std = scale_img(img_stack.std(0))
summaries = (img_corr, img_mean, img_max, img_std)

# get targets
[masks_soma, masks_border, masks_centroids] = \
    get_masks(labels_folder+suffixes[vid_num], collapse_masks=True, centroid_radius=3)
targets = (masks_soma, masks_border, masks_centroids)

## add label borders
border_color = (1, 0, 0)
summaries_border = [add_contours(x, masks_border, color=border_color) for x in summaries]

# display those guys
mosaic_summaries = np.concatenate(summaries, 1)
mosaic_targets = np.concatenate(targets, 1)
plt.subplot(2, 1, 1); plt.imshow(mosaic_summaries)
plt.subplot(2, 1, 2); plt.imshow(mosaic_targets)
plt.show()








