# ThirdEye 3D Pose Esimation ML

#### Latest Model -- `UNet-2D-Pose / Full_COCO_Train`
#### Look in WandB for all Model Training/Tracking Info

## File Structure
Directory  | Kept/Ignored | Details
------------- | -------------
Trainer  | Kept | Script and Config for Command Line Training
Lib  |  Kept | Support Functions and Model Definitions
Notebooks  |  Kept  | Trainable Notebooks to be Converted into /trainer for Real Training
Docs  | Kept  | Readable Docs w/ Useful Info
SavedModels  | Ignored | Kept in GCS Bucket gs://models-thirdeye
Data | Ignored  | Kept in GCS Bucket gs://training-data-thirdeye

## Trainer
Contains script to train model, config.py for parameters.
Code from latest notebook in functional form.

## SavedModels
Keras/tf Saved Models from Training 
-- Names/Dirs Coordinate w/ Weights & Biases.   

Add Models from gs://models-thirdeye to thirdeye-flask to serve them.   

## Docs
#### Tips - Tricks
Look here for implementation details for each version. Markdown files about troubleshooting and lessons learned.

## Notebooks
**02-2DPose-UNet-heatmap**  [WandB](https://app.wandb.ai/wjtaylor/unet-2d-pose)
- 2D Multi-Person Heatmap Pose (COCO Dataset). UNet sourced from GitHub implementation.

**01-resnet50-simpleRegression** - **Deprecated**  
- 2D Pose Regression for a Single Person (COCO Dataset). Resnet50 Decoder and a Single Dense Layer.

## Attributions
- [Towards Accurate Multi-person Pose Estimation in the Wild](https://arxiv.org/pdf/1701.01779.pdf)
-- Loss and Non-Max Suppression
- [UNet Implementation by zhixuhao](https://github.com/zhixuhao/unet)
-- UNet Keras Model Definition
