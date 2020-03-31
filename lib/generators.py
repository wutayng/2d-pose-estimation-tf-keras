"""
Data Generators for Thirdeye Models
"""
from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np
import random
from .utils import dataframe_to_mask


class h5_generator(Sequence) :
    """
    Data Generator for h5 Input 
    """
  
    def __init__(self, h5_data, batch_size, joint_input_indices, image_resolution, keypoint_radius) :
        """
        args:
            h5_data:             Read in h5 File
            batch_size:          Batch Size
            joint_input_indices: Indices of pd Keypoints DataFrames to Compute (Ignore Others)
            image_resolution:    Resolution of image (int)
            keypoint_radius:     Keypoint radius for Binary Classification Loss
        """
        # Define Input h5 data
        self.h5_data = h5_data
        # Define Batch Size
        self.batch_size = batch_size
        # Define Number of Batches in the h5 Data
        self.num_batches = int(len(list(h5_data.keys())) / self.batch_size)

        # Randomly Shuffle Key Names into Batches
        shuffled_keys = sorted(list(h5_data.keys()), key=lambda k: random.random())
        self.batches = []
        for b in range(0,len(shuffled_keys),self.batch_size):
              self.batches.append(shuffled_keys[b:b+self.batch_size])

        # Define Parameters for Heatmap Creation from Coordinates
        self.joint_input_indices = joint_input_indices
        self.image_resolution = image_resolution
        self.radius = keypoint_radius

        # Parameters for __next__ function
        # :src :https://stackoverflow.com/questions/54590826/generator-typeerror-generator-object-is-not-an-iterator/57101352
        self.n = 0
        self.max = self.__len__()
    
    def __len__(self) :
        """
        Return Number of Batches in Total Data
        """
        return self.num_batches
  
  
    def __getitem__(self, idx) :
        """
        Function for tensorflow to get a Batch of Data
        param idx: Index of batch in len(self.num_batches)
        return batch_x: Tensor of Images
        return batch_y: Tensor of Keypoints Binary Mask Images
        """
        # X Data, Img
        img_list = []
        # Y Data, Keypoints
        keypoints_list = []
        # Loop Through Batch
        for i in range(0, self.batch_size):
            img_list.append(self.h5_data[self.batches[idx][i]]['img'][:] / 255)
            df = pd.DataFrame(data=self.h5_data[self.batches[idx][i]]['keypoints'][:])
            keypoints_list.append(dataframe_to_mask(df, self.joint_input_indices, \
                                                    self.image_resolution, self.radius))

        # Stack X List into Tensor
        batch_x = np.stack(img_list, axis=0)
        # Stack Y List into Tensor
        batch_y = np.stack(keypoints_list, axis=0)

        return batch_x, batch_y

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result


    