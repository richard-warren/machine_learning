from keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt
import cv2
import ipdb as ipdb
import time





def show_predictions(X, Y, predictions, examples_to_show=3):
    
    
    # prepare figure
    channels = Y.shape[-1]
    fig, axes = plt.subplots(examples_to_show, channels+1, sharex=True, sharey=True)
    inds = np.random.choice(range(X.shape[0]), size=examples_to_show, replace=False)
    
    
    # plot ground true and predictions for each sample
    for row in range(examples_to_show):
        
        # get raw image
        raw_img = X[inds[row],:,:,0]
        
        # show ground truth
        labeled_img = add_labels_to_frame(raw_img, Y[inds[row]], range(channels))
        axes[row, 0].imshow(labeled_img)
        axes[row, 0].axis('off')    
                
        # show ground truth and predictions
        for channel in range(channels):
            labeled_img = add_labels_to_frame(raw_img, predictions[inds[row]], iter([channel]))
            axes[row, channel+1].imshow(labeled_img)
            axes[row, channel+1].axis('off')
    plt.tight_layout()





class DataGenerator(Sequence):
    # keras data generator class
    
    
    def __init__(self, img_inds, dataset, batch_size=16, shuffle=True, sample_weights=[], num_loss_fcns=1):
        # initialization
        
        self.img_inds = img_inds
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        if not len(sample_weights):
            sample_weights = np.ones(dataset.root.imgs.shape[0])
        self.sample_weights = sample_weights
        self.num_loss_fcns = num_loss_fcns
        
        self.img_dims = (dataset.root.imgs.shape[1], dataset.root.imgs.shape[2])
        self.channels = dataset.root.labels.shape[-1]
        
        self.on_epoch_end() # shuffle inds on initialization
        
        
    def __len__(self):
        # number of batches per epoch
        
        return int(np.floor(len(self.img_inds) / self.batch_size))
    
    
    def __getitem__(self, index):
        # gets data for batch
        
        # get data from h5 file
#        t = time.time()
        batch_inds = self.img_inds[index*self.batch_size : (index+1)*self.batch_size]
        X = self.dataset.root.imgs[batch_inds,:,:,:].astype('float32')
        Y = self.dataset.root.labels[batch_inds,:,:,:].astype('float32')
    
        # normalize
        for smp in range(len(batch_inds)):
            for channel in range(self.channels):
                if np.max(Y[smp,:,:,channel])>0:
                    Y[smp,:,:,channel] = Y[smp,:,:,channel] / np.max(Y[smp,:,:,channel])            
        X = X / 255
        
#        return X, Y, np.ones(Y.shape[0])
#        print('get batch time: %.2f' % (time.time()-t))
        return X, [Y for _ in range(self.num_loss_fcns)], [self.sample_weights[batch_inds] for _ in range(self.num_loss_fcns)] # return same Y and smp_weights multiple times if using intermediate supervision
    

    def on_epoch_end(self):
        # shuffle data at the end of epoch
        
        if self.shuffle:
            np.random.shuffle(self.img_inds)



def add_labels_to_frame(frame, labels, channels_to_show, whiskers=4):
    
    
    # get rgb values for each whisker
    channels = labels.shape[-1]
    cmap = plt.cm.jet
    colors = np.zeros((whiskers,3), dtype='float32')
    color_inds = np.linspace(0,1,whiskers) # not really inds, but evenly spaced values between zero and one
    for channel in range(whiskers):
        colors[channel,:] = cmap(color_inds[channel])[0:3]
    points = (channels-whiskers)/whiskers
    channel_color_inds = np.concatenate((np.arange(whiskers), np.repeat(list(range(whiskers)),points))) # e.g. with three whiskers and two points per whisker: [0 1 2 0 0 1 1 2 2]
                
    # get colored labels
    colored_labels = np.zeros((labels.shape[0], labels.shape[1], 3, channels), dtype='uint8')
    for channel in channels_to_show:
            colored_labels[:,:,:,channel] =  np.multiply(np.repeat(labels[:,:,channel,None], 3, axis=2), colors[channel_color_inds[channel],:])
    colored_labels = np.mean(colored_labels, axis=3) # collapse across all colors
    if np.max(colored_labels)>0:
        colored_labels = colored_labels * (255/np.max(colored_labels))
#    ipdb.set_trace()
    # upsample labels if necessary
    if not frame.shape==labels.shape:
        colored_labels = cv2.resize(colored_labels, (frame.shape[1], frame.shape[0]))
        
    # merge frame with colored labels
    frame = np.repeat(frame[:,:,None], 3, axis=2) # add color dimension to frame
    merged = np.clip(cv2.addWeighted(colored_labels.astype('uint8'), 1.0, frame, 1.0, 0), 0, 255) # overlay raw image
    
    return merged


def add_maxima_to_frame(frame, labels, channels_to_add, whiskers=4):
    
    
    # get rgb values for each whisker
    channels = labels.shape[-1]
    cmap = plt.cm.jet
    colors = np.zeros((channels,3), dtype='float32')
    color_inds = np.linspace(0,1,whiskers) # not really inds, but evenly spaced values between zero and one
    for channel in range(whiskers):
        colors[channel,:] = cmap(color_inds[channel])[0:3]
    points = (channels-whiskers)/whiskers
    channel_color_inds = np.concatenate((np.arange(whiskers), np.repeat(list(range(whiskers)),points))) # e.g. with three whiskers and two points per whisker: [0 1 2 0 0 1 1 2 2]
    
    
    plt.close('all')
    plt.ioff() # turn off interactive mode to prevent figures from displaying
    fig = plt.figure(frameon=False, figsize=(frame.shape[1]*.01,frame.shape[0]*.01))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.imshow(frame)
    plt.axis('off')
        
    for chan in channels_to_add:
        x, y = np.unravel_index(np.argmax(labels[:,:,chan]), labels.shape[0:2])
        # upsample labels if necessary
        if not frame.shape==labels.shape:
            x, y = x*(frame.shape[0]/labels.shape[0]), y*(frame.shape[1]/labels.shape[1])
        plt.plot(y, x, 'o', ms=10, alpha=0.8, color=colors[channel_color_inds[chan],:])
    
    fig.canvas.draw()
    frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    w, h = fig.canvas.get_width_height()
    frame = frame.reshape((h, w, 3))
    
    return frame



