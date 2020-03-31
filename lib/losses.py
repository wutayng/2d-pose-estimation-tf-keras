"""
Losses for Thirdeye Models
"""
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow import cast, greater, where, equal, constant, expand_dims
from tensorflow import float64, int64

import numpy as np
from scipy.ndimage.filters import maximum_filter

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
        y_true_binary = cast(greater(y_true,0), float64)
        # Loss Calculation
        loss_bce = binary_crossentropy(y_true_binary, y_pred)
        
        return loss_bce
    
    return bce_logisticloss


def normKPM(y_true, y_pred):
    """
    Return Normalized Keypoint Precision Metric Calculated from Predicted Heatmaps
    Joint Coordinate Error Expressed as a Percentage of Screen Resolution
    
    Using Non-Max Supression Keypoint Detection (OpenPose Method)
        Scipy Maximum Filter (scipy.ndimage.filters.maximum_filter)
        src: https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-2-e78ab9104fc8
    """
    
    # Initialize Noramlized Precision List
    precision_list = []
    
    # For Each Batch
    for batch in range(y_true.shape[0]):
            
        # For Each Heatmap in Batch
        for heatmap in range(y_true.shape[3]):
    
            # Load Images From Loop
            y_metric_true = y_true[batch, :, :, heatmap]
            y_metric_pred = y_pred[batch, :, :, heatmap]

            # True Keypoints
            keypoints_loc = where(equal(y_metric_true,1))
            
            # Non-Max Supression
            window_size = 50
            keypoint_candidates = y_metric_pred*(y_metric_pred == maximum_filter(y_metric_pred,
                                                                       footprint=np.ones((window_size,window_size))
                                                                        ))
            keypoints_pred_loc = where(greater(keypoint_candidates,0))

            # If not Enough Keypoints Exist, Add [0,0] Keypoints
            keypoint_zeros = expand_dims(constant([0, 0],dtype=int64), axis=0)
            if keypoints_loc.shape[0] > keypoints_pred_loc.shape[0]:
                for _ in range(keypoints_loc.shape[0] - keypoints_pred_loc.shape[0]):
                    keypoints_pred_loc = np.concatenate((keypoints_pred_loc, keypoint_zeros), axis=0)

            # Select N Maximum Heatmap Values Keypoint Candidates
            # Candidate_selection is a List of Row Indexes Corresponding to Keypoints_Pred_loc
            rand_selection = np.random.choice(np.arange(0, keypoints_pred_loc.shape[0], 1),
                                              replace=False,
                                              size=(3),)
            heatmap_values = []
            for i in range(keypoints_pred_loc.shape[0]):
                heatmap_values.append(y_metric_pred[keypoints_pred_loc[i,0],keypoints_pred_loc[i,1]])  

            candidate_selection = []
            for i in range(keypoints_loc.shape[0]):
                curr_max = np.amax(np.asarray(heatmap_values))
                curr_index = np.where(curr_max == np.asarray(heatmap_values))[0][0]
                candidate_selection.append(curr_index)
                heatmap_values = np.delete(heatmap_values, curr_index)
                
            # Find Pixel Euclidian Distance From Each Real Keypoints
            euclidian_distances_list = []
            for i in range(len(candidate_selection)):   
                distances = []
                for k in range(keypoints_loc.shape[0]):
                    distances.append(np.linalg.norm(keypoints_pred_loc[candidate_selection[i],:]-keypoints_loc[k,:]))
                euclidian_distances_list.append(np.reshape(np.asarray(distances),(keypoints_loc.shape[0],1)))
                
            # If Keypoint Exists
            if keypoints_loc.shape[0] > 0:
                euclidian_distances = np.stack(euclidian_distances_list,axis=2)
                
                # Find Average Distance Norm
                sum_distances = 0
                for k in range(keypoints_loc.shape[0]):
                    sum_distances += np.amin(euclidian_distances[k,:])

                # Norm is Percentage of Resolution
                precision_list.append((sum_distances/keypoints_loc.shape[0])/y_metric_pred.shape[0])
                
            # If Keypoints Doesn't Exist, Don't Append Precision_list
            else:
                pass
    
    # Return Average Precision Converted to Percentage of Resolution
    return (np.average(precision_list) * 100)

    