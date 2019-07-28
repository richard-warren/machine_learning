# cell's kitchen
neuron segmentation for calcium imaging. to be used as initialization for CalmAn

**stage 1 - region proposal**  
identifies potential neuron locations using fully convolutional networks to segment neurons from background (region)

**stage 2 - instance segmentation**  
applies a segmentation network to subframes centered at maxima of segmentation from stage 1, yielding masks for individual neurons

##instructions

#### setup
make local data directory containing training videos and labels. the directory should have the following subdirectories:
   * datasets: contains folders for each training video, with folders named 'images_J115', 'image_J123', etc.
   * labels: contains folders with labels for each each training video, with format 'J115', J123', etc.   

edit region_proposal/config.py
   * data_dir: change to reflect the location of the data directory
   * datasets: change to reflect the datasets in the data directory
   * test_datasets, train_datasets: datasets to be used for training and testing/validation
   * X_layers, y_layers: choose which summary and target images to include in the network

## prepare training data
in command line from cells_kitchen root directory, prepare training data as follows:
```
python region_proposal\prepare_training_data.py
```
this creates training_data folder in your data directory that contains numpy files with summary images and targets that are used for training
## train!
```
python region_proposal\train.py
```
this will start training the network, saving models in a 'models' folder in your data directory. each model has it's own unique subfolder with the data and time.
during training predictions are generated and saved as images in the model folder so you can see how the predictions are evolving with training

losswise.com can be used for nice visualizations during training. you can enter your losswise_api_key in the config file to visualize training through your losswise account


