"""
Config for train.py
"""
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.metrics import Precision, Recall

H5_TRAIN_FILE = '../data/coco_train_256_x56637.h5'
H5_VAL_FILE = '../data/coco_val_256_x2366.h5'

USE_GENERATOR = True
SAVE_MODEL = True # Whether or Not to Save the Model Output
WANDB = True # Boolean whether or not to log in Weights and Biases

PROJECT_NAME = 'unet-2d-pose'
RUN_NAME = 'Full_COCO_Train'
NOTES = 'Full COCO Dataset Training'
MODEL = 'UNet'
DATA = 'COCO'

BATCH_SIZE = 12
KEYPOINT_RADIUS = 5 # Keypoint Radius for Binary Classification Loss
NUM_SAMPLES = 56636 # Max = 56636, From h5 File
NUM_VAL_SAMPLES = 2366 # Max = 2366, From h5 File
EPOCHS = 12
IMAGE_RESOLUTION = 256 # Image Shape (int)

LOSS_TYPE ='bce_logisticloss' # Write str of Loss Function Used
METRICS_THRESHOLD = 0.05
METRICS = [Precision(thresholds=METRICS_THRESHOLD), Recall(thresholds=METRICS_THRESHOLD)]
LEARNING_RATE = 'Adam Standard'
DECAY = 'Adam Standard'
OPTIMIZER = Adam()

VMIN = 0.01 # Colormap Lower Cutoff for Displaying Heatmaps
CMAP = plt.get_cmap('hsv')
CMAP.set_under('k', alpha=0)

DESIRED_JOINTS = ['left_shoulder',
                  'right_shoulder',
                  'left_elbow',
                  'right_elbow',
                  'left_wrist',
                  'right_wrist',
                  'left_hip',
                  'right_hip',
                  'left_knee',
                  'right_knee',
                  'left_ankle',
                  'right_ankle'
                 ]

def init_wandb():
    wandb.init(project=PROJECT_NAME,
               notes=NOTES,
               name=RUN_NAME,
               resume=False,
               )
    wandb.config.update({
        "model": MODEL,
        "data": DATA,
        "use_generator": USE_GENERATOR,
        "img_resolution": IMAGE_RESOLUTION,
        "loss": str(LOSS_TYPE),
        "metrics": METRICS,
        "metrics_threshold": METRICS_THRESHOLD,
        "samples": NUM_SAMPLES,
        "val_samples": NUM_VAL_SAMPLES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": str(OPTIMIZER),
        "lr": str(LEARNING_RATE),
        "decay": str(DECAY),
        "keypoint_radius": KEYPOINT_RADIUS,
        })
    wandb_callback = WandbCallback(
            save_model=False,
            monitor='loss',
            mode='auto',
            )
    return wandb_callback
