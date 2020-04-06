"""
Metrics for Thirdeye Models
"""
from tensorflow import nn, cast, greater, equal, where, zeros_like
from tensorflow import float32, int32

from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K


class NMS(Metric):
    """
    UNFINISHED
    Fragment Left Here for Future Work
    """
    def __init__(self, window_size, nms_threshold, name='NMS'):
        """
        args:
            window_size: Non-Maximum Supression Window Size in Pixels
            nms_threshold: Non-Max Suppression Threshold, 0-1 Scale
        """
        self.__name__ = name
        self.window_size = window_size
        self.nms_threshold = nms_threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        # True Keypoints
        true = cast(equal(y_true,1),int32)

        # Non-Max Supression
        pooled = nn.max_pool2d(y_pred, ksize=[1,self.window_size,self.window_size,1],
                                      strides= [1,1,1,1], padding="SAME")
        nms_pred = where(equal(y_pred, pooled), y_pred,
                           zeros_like(y_pred,dtype=float32))
        pred = cast(greater(nms_pred,self.nms_threshold),int32)

        # Precision
        
        # UNFINISHED -- NEEDS TO BE UPDATED
        
        self.nms = 1

    def result(self):
        return self.nms

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.nms.assign(0.)