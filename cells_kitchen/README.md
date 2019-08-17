# cell's kitchen (work in progress)
neuron segmentation for calcium imaging. to be used as initialization for CalmAn

**stage 1 - region proposal**  
identifies potential neuron locations using fully convolutional networks to segment neurons from background

**stage 2 - instance segmentation**  
applies a segmentation network to subframes centered at maxima of segmentation from stage 1, yielding masks for individual neurons as well as confidence scores for each subframe

## instructions

### setup
start by adding the directory containing 'cells_kitchen' to PYTHONPATH (todo: should i use pip install -e for this instead???)

make local data directory containing training videos and labels. the directory should have the following subdirectories:
   * datasets: contains folders for each training video, with folders named 'images_J115', 'image_J123', etc.
   * labels: contains folders with labels for each each training video, with format 'J115', J123', etc.   
   
### prepare training data
edit config.py
   * data_dir: change to reflect the location of the data directory
   * datasets: change to reflect the datasets in the data directory
   
in command line from cells_kitchen root directory, prepare training data as follows:
```
python prepare_training_data.py
```
this creates training_data folder in your data directory that contains numpy files with summary images and targets that are used for training.
the folder will also contain .png files showing all summary and target images for reference.
### train!
train region proposal network after editing region_proposal\config.py:
```
python region_proposal\train.py
```
this will start training the network, saving models in a 'region_proposal\models' folder in your data directory. each model has it's own unique subfolder with the date and time.
during training predictions are generated and saved as images in the model's folder so you can see how the predictions are evolving with training

next, train the segmentation network:
```
python instance_segmentation\train.py
```
trained models will appear in instance_segmentation\models

