# UNet 2D Monocular RBG Pose Esimation

#### Finished Model -- `Full_COCO_Train`

#### Look in [WandB](https://app.wandb.ai/wjtaylor/unet-2d-pose?workspace) for all Model Training / Tracking Info

### Example Output Heatmaps

<div><img width="450" src="https://github.com/wutayng/2d-pose-estimation-tf-keras/blob/master/assets/heatmap-inference.png" />
<img width="400" src="https://github.com/wutayng/2d-pose-estimation-tf-keras/blob/master/assets/beach_running_inference.gif" />
</div>

<div><img src="https://github.com/wutayng/2d-pose-estimation-tf-keras/blob/master/assets/UNet-2DPose_WandB.png" /></div>

## To serve the model here (/SavedModels) via a flask web API, check out [flask-keras-server](https://github.com/wutayng/flask-keras-server)

## Process-coco

Here is a script to convert COCO images and annotations into h5 files containing the images and keypoint annotation coordinates. h5 files must be created to use anything else in this repo (besides the SavedModel)!

## Trainer

```
python train.py
```

Script to train model, edit config.py for parameters.
The training script copies code from the notebook in functional form - put the h5 COCO data files in /data to train!

## SavedModels

Contains a Keras Saved Model. UNet 2D Pose trained on the full COCO Dataset.

## Docs

#### Tips - Tricks

Look here for some more implementation details about troubleshooting and lessons learned.

## Notebooks

**02-2DPose-UNet-heatmap** [WandB](https://app.wandb.ai/wjtaylor/unet-2d-pose?workspace)

-   2D Multi-Person Heatmap Pose (COCO Dataset). UNet sourced from GitHub implementation. Create the h5 COCO data files via /process-coco and place in /data to train!

## Attributions

-   [COCO Dataset](http://cocodataset.org/)
-   [Towards Accurate Multi-person Pose Estimation in the Wild](https://arxiv.org/pdf/1701.01779.pdf)
    -- Loss and Non-Max Suppression
-   [UNet Implementation by zhixuhao](https://github.com/zhixuhao/unet)
    -- UNet Keras Model Definition
