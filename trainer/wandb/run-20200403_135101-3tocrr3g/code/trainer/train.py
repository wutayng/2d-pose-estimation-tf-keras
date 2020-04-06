"""
Train Script Created From Latest Notebook
Current Version - 02-2DPose_UNet
"""
import os, sys, h5py, wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import save_model
import tensorflow as tf

# Relative Imports
wk_dir = os.path.split(os.getcwd())[0]
print(wk_dir)
if wk_dir not in sys.path:
    sys.path.append(wk_dir)
from config import *
from lib import utils, generators, losses, metrics
from lib.models.unet_simple import unet_simple

            
def define_h5():
    # Load h5 File
    h5_train_data = h5py.File(H5_TRAIN_FILE, 'r')
    h5_val_data = h5py.File(H5_VAL_FILE, 'r')

    # Joints List from h5 Input
    input_joints = list(h5_train_data.attrs['jointnames'])
    # Create List of Original Input Indices
    joint_input_indices = []
    for i in input_joints:
        for d in DESIRED_JOINTS:
            if i == d:
                joint_input_indices.append(input_joints.index(i))

    return h5_train_data, h5_val_data, joint_input_indices


def init_model():
    # Create Model
    model = unet_simple(input_shape = (IMAGE_RESOLUTION, IMAGE_RESOLUTION, 3), \
                     output_num = len(DESIRED_JOINTS))
    model.compile(optimizer = OPTIMIZER, loss = losses.customLoss(), metrics = METRICS)
    return model


def load_generators():
    if USE_GENERATOR == True:
        batch_train = BATCH_SIZE
        batch_val = BATCH_SIZE
    if USE_GENERATOR == False:
        batch_train = NUM_SAMPLES
        batch_val = NUM_VAL_SAMPLES
    
    h5_train_data, h5_val_data, joint_input_indices = define_h5()
    
    training_generator = generators.h5_generator(h5_train_data, batch_size=batch_train,
                                  joint_input_indices=joint_input_indices,
                                  image_resolution=IMAGE_RESOLUTION, 
                                  keypoint_radius=KEYPOINT_RADIUS)
    val_generator = generators.h5_generator(h5_val_data, batch_size=batch_val,
                                  joint_input_indices=joint_input_indices,
                                  image_resolution=IMAGE_RESOLUTION, 
                                  keypoint_radius=KEYPOINT_RADIUS)
    
    return training_generator, val_generator


def train(CALLBACKS):
    
    training_generator, val_generator = load_generators()
    
    model = init_model()
    
    # Train Model
    if USE_GENERATOR == True:
        model.fit_generator(
                            generator=training_generator,
                            steps_per_epoch=int(NUM_SAMPLES/BATCH_SIZE),
                            validation_data=val_generator,
                            validation_freq=1,
                            validation_steps=int(NUM_VAL_SAMPLES/BATCH_SIZE),
                            epochs=EPOCHS,
                            callbacks=CALLBACKS,
                            )

    elif USE_GENERATOR == False:
        # Train Model
        model.fit(x=training_generator.load_data(0,0), y=training_generator.load_data(0,1), batch_size=BATCH_SIZE, 
              epochs=EPOCHS, callbacks=CALLBACKS,
              validation_data=val_generator.__getitem__(0),
              shuffle=True, verbose=1,
                 )
    
    return model


def predict_sample(model):
    # Predict and Log Plot to WandB
    
    h5_train_data, h5_val_data, joint_input_indices = define_h5()
    
    # Get Random Index in h5 File
    pred_generator = generators.h5_generator(h5_val_data, batch_size=1,
                                      joint_input_indices=joint_input_indices,
                                      image_resolution=IMAGE_RESOLUTION, 
                                      keypoint_radius=KEYPOINT_RADIUS)
    pred_img_tensor = pred_generator.load_data(0,0)

    # Predict Image Joints from Sample
    predictions = model.predict(pred_img_tensor, verbose=1)
    
    # Overlay all Joints on Original Image
    plt.imshow(pred_img_tensor[0,:])
    for keypoint in range(len(DESIRED_JOINTS)):
        # If Keypoint Exists in image
        if np.amax(predictions[0, :, :, keypoint]) != 0:
            plt.imshow(predictions[0, :, :, keypoint], cmap=CMAP, vmin=VMIN)

    plt.gcf().set_size_inches(6, 6)
    
    print('Max Val of Prediction Tensor: {}'.format(tf.math.reduce_max(predictions)))

    if WANDB == True:
        wandb.log({"Full Heatmap": plt})
        
    return
    

if __name__ == "__main__":
    CALLBACKS = []
    if WANDB == True:
        wandb_callback = init_wandb()
        CALLBACKS.append(wandb_callback)
        
    model = train(CALLBACKS)
    
    predict_sample(model)
    
    if SAVE_MODEL == True:
        save_model(
            model, ('../SavedModels/' + PROJECT_NAME + '/' + RUN_NAME), overwrite=True, include_optimizer=True,
            signatures=None, options=None
        )