"""
Train Script Created From Latest Notebook
Current Version - 02-2DPose_UNet

No Training Generator, Load Full Data into RAM
"""
import os, sys, h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import save_model

# Memory Benchmarking
import psutil
process = psutil.Process(os.getpid())

# Relative Imports
wk_dir = os.path.split(os.getcwd())[0]
print(wk_dir)
if wk_dir not in sys.path:
    sys.path.append(wk_dir)
    
from config import *
from lib import utils, generators, losses
from lib.models.unet_simple import unet_simple

# Callbacks
class Metrics_Callback(Callback):

    def __init__(self, val_data):
        self.validation_data = val_data

    def on_train_begin(self, logs=None):
        self.val_normKPMs = []

    def on_epoch_end(self, epoch, logs=None):

        # Predict Validation Data
        prediction_val = self.model.predict(self.validation_data[0])
        
        # Perform norm_KPM Calculation on Validation Data
        normKPM = losses.normKPM(self.validation_data[1], prediction_val)
        print('\n normKPM Value: {} \n'.format(normKPM))
        
        if epoch == 0:
            print('Total Memory Allocation During Training: {} \n'.format(
                process.memory_info().rss/1000000000) ) # in gigabytes 
        
        if WANDB == True:
            wandb.log({'val_normKPMs': normKPM}, step=epoch)

# WandB
if WANDB == True:
    wandb_callback = init_wandb()

# Load h5 File
h5_train_data = h5py.File('../data/coco_train_256_x56637.h5', 'r')
h5_val_data = h5py.File('../data/coco_val_256_x2366.h5', 'r')

# Joints List from h5 Input
input_joints = list(h5_train_data.attrs['jointnames'])
# Create List of Original Input Indices
joint_input_indices = []
for i in input_joints:
    for d in desired_joints:
        if i == d:
            joint_input_indices.append(input_joints.index(i))

# Create Model
model = unet_simple(input_shape = (IMAGE_RESOLUTION, IMAGE_RESOLUTION, 3), \
                     output_num = len(desired_joints))
model.compile(optimizer = OPTIMIZER, loss = losses.customLoss(), metrics = METRICS)

# Use Generator to Load All Data into Tensors
training_generator = generators.h5_generator(h5_train_data, batch_size=NUM_SAMPLES,
                              joint_input_indices=joint_input_indices,
                              image_resolution=IMAGE_RESOLUTION, 
                              keypoint_radius=KEYPOINT_RADIUS)
[ img_train_tensor, keypoints_train_tensor ] = training_generator.__getitem__(0)
    
val_generator = generators.h5_generator(h5_val_data, batch_size=NUM_VAL_SAMPLES,
                              joint_input_indices=joint_input_indices,
                              image_resolution=IMAGE_RESOLUTION, 
                              keypoint_radius=KEYPOINT_RADIUS)
[ img_val_tensor, keypoints_val_tensor ] = val_generator.__getitem__(0)

CALLBACKS = [Metrics_Callback(val_data=[ img_val_tensor.astype(float), keypoints_val_tensor.astype(float) ])]
if WANDB == True:
    CALLBACKS.append(wandb_callback)
        
print('Total Memory Allocation Before Training: {}'.format(
    process.memory_info().rss/1000000000) ) # in gigabytes 


# Train Model
model.fit(x=img_train_tensor, y=keypoints_train_tensor, batch_size=BATCH_SIZE, 
          epochs=EPOCHS, callbacks=CALLBACKS,
          validation_data=[img_val_tensor, keypoints_val_tensor],
          shuffle=True, verbose=1,
          )


# Predict and Log Plot
# Get Random Index in h5 File
pred_generator = generators.h5_generator(h5_val_data, batch_size=1,
                                  joint_input_indices=joint_input_indices,
                                  image_resolution=IMAGE_RESOLUTION, 
                                  keypoint_radius=KEYPOINT_RADIUS)
[ pred_img_tensor, pred_keypoints_tensor ] = pred_generator.__getitem__(0)
# Predict Image Joints from Sample
predictions = model.predict(pred_img_tensor, verbose=1)
# Overlay all Joints on Original Image
plt.imshow(pred_img_tensor[0,:])
for keypoint in range(len(desired_joints)):
    # If Keypoint Exists in image
    if np.amax(predictions[0, :, :, keypoint]) != 0:
        plt.imshow(predictions[0, :, :, keypoint], cmap=CMAP, vmin=VMIN)
    
plt.gcf().set_size_inches(6, 6)

if WANDB == True:
    wandb.log({"Full Heatmap": plt})

print('wow')