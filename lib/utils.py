"""
Utilities for Thirdeye Models
"""
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd

    
def dataframe_to_heatmap(df, joint_indices, resolution, std=1):
    """
    Convert pd DataFrame of Keypoint Coordinates to Heatmap Masks

    df - keypoint coordinate dataframe, rows=coord1(x),coord1(y),coord2(x)..., cols=person1,person2,...
    joint_indices - indices of joints to compute (via joint_names list)
    resolution - image resolution (int)
    std - standard deviation of heatmap keypoint gaussian kernel

    """
    heatmap_list = []
    # For Each Joint in DataFrame
    for joint in joint_indices:
        # Initialize Heatmap Grid
        x = np.linspace(0, int(resolution/std), resolution)
        y = np.linspace(0, int(resolution/std), resolution)
        xx, yy = np.meshgrid(x,y)
        # Evaluate all Kernels at Grid and Sum
        xxyy = np.c_[xx.ravel(), yy.ravel()]
        # Preallocate Probability Density Output
        zz = np.zeros((resolution*resolution))

        # For Each Person in Image
        for person in range(0,len(df.columns),2):
            # If Keypoint is Nonzero
            if df.iloc[joint][person] != 0:
                k = multivariate_normal(mean=(df.iloc[joint][person]/std,\
                                               df.iloc[joint][person+1]/std), cov=np.eye(2))
                # Add Keypoint to Kernel Grid
                zz += k.pdf(xxyy)

        # Rescale zz to 0-1 if Nonzero
        if np.amax(zz) != 0:
            zz = zz * (1/np.amax(zz))

        # Reshape into Output Heatmap Mask (Round to 3 Decimal)
        heatmap_list.append(np.round(zz.reshape((resolution,resolution)), 3))

    # Stack Keypoints List into Channels
    return np.stack(heatmap_list, axis=2)


