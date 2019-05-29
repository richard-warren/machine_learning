## initializations
prefix = "F:\\cells_kitchen_files\\datasets\\images_"
labels_folder = "F:\\cells_kitchen_files\\labels\\"
suffixes = ['N.00.00', 'N.01.01', 'N.02.00', 'N.03.00.t', 'N.04.00.t', 'YST']

## look at some sweet, sweet vids
vid_num = 0

preview_vid(prefix+suffixes[vid_num], frames_to_show=50, fps=100)

## show summary and target images for vid
vid_num = 0
frames = 200

# get summary images
img_stack = get_frames(prefix+suffixes[vid_num], frames, contiguous=False)
img_corr = get_correlation_image(img_stack)
img_median = np.median(img_stack, 0)
img_max = img_stack.max(0)
img_std = scale_img(img_stack.std(0))
summaries = (img_corr, img_median, img_max, img_std)

# get targets
[masks_soma, masks_border, masks_centroids] = \
    get_masks(labels_folder+suffixes[vid_num], collapse_masks=True, centroid_radius=3)
targets = (masks_soma, masks_border, masks_centroids)

## add contours
border_color = (.5, 0, 0)
summaries_borders = [add_contours(x, masks_border, color=border_color) for x in summaries]


## display those guys
mosaic_summaries = np.concatenate(summaries_borders, 1)
mosaic_targets = np.concatenate(targets, 1)
plt.imshow(mosaic_summaries)
# plt.imshow(mosaic_targets)
plt.show()

##












