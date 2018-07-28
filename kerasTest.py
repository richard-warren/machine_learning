# imports
import numpy as np
np.random.seed(123) # for reproducibility (how?)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import MaxPooling2D, Convolution2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt



# load hand-written numbers
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape input data to have depth of 1
imgDims = X_train.shape[1:3]
categories = max(y_test) + 1
X_train = X_train.reshape(X_train.shape[0], imgDims[0], imgDims[1], 1)
X_test = X_test.reshape(X_test.shape[0], imgDims[0], imgDims[1], 1)

# change to float32 and normalize
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# convert labels to one-hot
y_train = np_utils.to_categorical(y_train, categories)
y_test = np_utils.to_categorical(y_test, categories)




# construct network
model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(imgDims[0],imgDims[1],1)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(categories, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



# train! omg!
model.fit(X_train, y_train, 
          batch_size=32, epochs=10, verbose=1)

# test model! omg!
predicted_classes = model.predict(X_test)
predicted_classes = predicted_classes.argmax(axis=1)
score = model.evaluate(X_test, y_test, verbose=1)

# show numbers predicted to belong to a single class
num = 1
imgs_to_show = 5
single_class_imgs = np.squeeze(X_test[predicted_classes==num])
imgs_concatenated = np.reshape(single_class_imgs[0:imgs_to_show], (imgs_to_show*imgDims[0], imgDims[1]))
plt.imshow(imgs_concatenated)





