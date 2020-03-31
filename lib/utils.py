"""
Utilities for Thirdeye Models
"""
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd

def dataframe_to_mask(df, joint_indices, resolution, radius):
    """
    Convert pd DataFrame of Keypoint Coordinates to Binary Pixel Masks (Radius R Around Keypoint)
    Center of Keypoint (Ground Truth) = 1. Keypoints in Radius = 0.9

    args:
        df:            Keypoint coordinate dataframe, rows=coord1(x),coord1(y),coord2(x)..., cols=person1,person2,...
        joint_indices: Indices of joints to compute (via joint_names list)
        resolution:    Image resolution (int)
        radius:        Radius of Keypoint (In Pixels) set to =0.9 for Binary Classification
        
    return:
        mask_array:    Resolution x Resolution np.array mask
    """
    mask_list = []
    # For Each Joint in DataFrame
    for joint in joint_indices:
        
        # Create Zeros Mask Image
        mask = np.zeros((resolution,resolution))
        
        # For Each Person in Image
        for person in range(0,len(df.columns),2):
             # If Keypoint is Nonzero
            if df.iloc[joint][person] != 0:
                # Y Coordinate is Rows, X is Cols
                # Offset -1 for Pixel Coordinate to Pixel Index
                x_coord = int(np.around(df.iloc[joint][person])) - 1
                y_coord = int(np.around(df.iloc[joint][person+1])) - 1
                
                # Set Radius Pixels to 0.9
                # For Loop through Possible Radius Values
                for radius_x in range(-radius,radius+1):
                    for radius_y in range(-radius,radius+1):
                        # If Current Pixel Location is Within Radius, Set Value
                        if ((radius_x**2)+(radius_y**2)) <= (radius**2):
                            # Make Sure Pixel is Within Image Coordinate Bounds
                            if 0 <= (y_coord+radius_y) < (resolution) and 0 <= (x_coord+radius_x) < (resolution):
                                mask[y_coord+radius_y][x_coord+radius_x] = 0.9
                
                # Set True Keypoint Pixel to 1
                mask[y_coord][x_coord] = 1

        # Append List with Joint Mask
        mask_list.append(mask)

    # Stack Keypoints Masks into Tensor
    return np.stack(mask_list, axis=2)


