# ThirdEye 3D Pose Esimation ML

## Contents:

### Lib
Training Support Functions.  
Data Generation, Loss Function, and Heatmap Utils.    

### Notebooks
02-2DPose-UNet-heatmap  
- 2D Multi-Person Heatmap Pose (COCO Dataset). UNet sourced from GitHub implementation.

01-resnet50-simpleRegression - Deprecated  
- 2D Pose Regression for a Single Person (COCO Dataset). Resnet50 Decoder and a Single Dense Layer.

### Data
Ignored from git for obvious reasons. Copy from GCS for training.

### SavedModels
Keras/TF Saved Models - Names/Dirs Coordinate w/ Weights & Biases for Tracking.   
Ignored from git. Copied to GCS Bucket models-thirdeye.  
Add Models to thirdeye-flask to serve them.   