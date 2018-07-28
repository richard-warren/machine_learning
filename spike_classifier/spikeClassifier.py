# data augmentation // split train and test, ensuring even split for smp and cpx spikes // try early stopping // try resnet structure

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Conv1D
from keras import backend as K


# settings
testPortion = .2
sensitivity_over_specificity = 0.9

# prepare input data
spikeData = loadmat('spikeData.mat')
sSpks = spikeData['sple_waveform']
cSpks = spikeData['cplx_waveforms']
x = np.concatenate((sSpks, cSpks), axis=0)
x = x.reshape(x.shape[0], x.shape[1], 1) # add a depth dimension
x = (x-np.mean(x)) / np.std(x)
dim = x.shape[1]

# prepare target data
labels = np.zeros((x.shape[0],1))
labels[sSpks.shape[0]:]=1
labels = np_utils.to_categorical(labels, 2)

# split into train and test sets
x_train, x_test, labels_train, labels_test = train_test_split(
        x, labels, test_size=testPortion)


# construct network
model = Sequential()
#model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(32, 10, activation='relu', input_shape=(dim,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(64, 10, activation='relu', input_shape=(dim,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(128, 10, activation='relu', input_shape=(dim,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# define training metrics
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


# train network
model.summary()
class_weights = {0: 1, 1: (sSpks.shape[0]/cSpks.shape[0]) * sensitivity_over_specificity}
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[sensitivity, specificity])
model.fit(x_train, labels_train, 
          batch_size=32, epochs=10, verbose=1, class_weight=class_weights)

# test network
predicted_labels = model.predict(x_test)
predicted_labels = predicted_labels.argmax(axis=1) == 1
score = model.evaluate(x_test, labels_test, verbose=1)
print('test accuracy:', score[1])
print('test sensitivity:', np.sum(labels_test[:,1]*predicted_labels) / np.sum(labels_test[:,1]))
print('test specificity:', np.sum(labels_test[:,0]*(predicted_labels==0)) / np.sum(labels_test[:,0]))


# show examples
rows = 6
cols = 8


figTitles = ['simple spikes', 'complex spikes']
plt.close('all')
for spkType in range(2):
    
    inds = np.array(np.where(labels_test[:,1]==spkType))
    inds = np.random.choice(np.squeeze(inds), size=rows*cols, replace=True)
    
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
    indsInd=0
    for row in range(rows):
        for col in range(cols):
            color='blue' if predicted_labels[inds[indsInd]]==labels_test[inds[indsInd],1] else 'red'
            ax = axes[row,col].plot(range(dim),
                     x_test[inds[indsInd],:], color=color)
            axes[row,col].axis('off')
            indsInd+=1
    fig.suptitle(figTitles[spkType])
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()


