

img_num = 0

frame = file.root.imgs[img_num,:,:,0]
labels = file.root.labels[img_num,:,:,:]

from utils import add_labels_to_frame
labelled = add_labels_to_frame(frame, labels, iter([16]), whiskers=4)
plt.imshow(labelled)