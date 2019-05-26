##


img_num = 0


##


frame = file.root.imgs[img_num,:,:,0]
labels = file.root.labels[img_num,:,:,:]
## todo:
from utils import add_labels_to_frame
labeled = add_labels_to_frame(frame, labels, range(12), whiskers=4)
plt.imshow(labeled)


##

b = 2



