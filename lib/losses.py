"""
Losses for Thirdeye Models
"""
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow import cast, greater
from tensorflow import float32

import numpy as np

def customLoss():
    """
    Custom Loss Wrapper to Input Arguments
    args:
        null
    """
    def bce_logisticloss(y_true, y_pred):
        """
        Based on:
        Ref: https://arxiv.org/pdf/1701.01779.pdf
        Google's Custom Loss from 'Towards Accurate Multi-person Pose Estimation in the Wild'
        ONLY USING CLASSIFICATION LOSS COMPONENT

        Sum Logistic Loss Based on Binary Classification
        For Heatmap Representation, Convert Heatmap to Binary Mask via Threshold Value
        """
        ## CLASSIFICATION COMPONENT
        # Set all Non-Zero Values in Mask to 1
        y_true_binary = cast(greater(y_true,0), dtype=float32)
        # Loss Calculation
        loss_bce = binary_crossentropy(y_true_binary, y_pred)
        
        return loss_bce
    
    return bce_logisticloss


    