



locations = np.array([25, 50])

label = np.empty(img_dims, dtype='float32') # set confidence map to all zeros if whisker is not in frame


X, Y = np.meshgrid(range(img_dims[1]), range(img_dims[0]))
temp = np.exp(-np.sqrt((np.power(Y-locations[1],2) + np.power(X-locations[0],2))) / (2*label_filtering^2))



