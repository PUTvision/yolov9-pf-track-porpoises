import cv2
import numpy as np
import copy


# it is based on GMC from BoT-SORT
class SparseOpticalFlow:
    def __init__(self) -> None:
        self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3,
                                       useHarrisDetector=False, k=0.04)
        
        self.downscale_factor = 2.0
        
        self.prev_frame = None
        self.prev_keypoints = None


    def update(self, frame: np.ndarray, frame_index: int):
        height, width, _ = frame.shape
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)
        
        if self.downscale_factor > 1.0:
            frame_gray = cv2.resize(frame_gray, (int(width / self.downscale_factor), int(height / self.downscale_factor)))
            
        keypoints = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
        
        if self.prev_frame is None or self.prev_keypoints is None:
            self.prev_frame = frame_gray.copy()
            self.prev_keypoints = copy.copy(keypoints)
            
            return H
        
        matched_keypoints, status, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, frame_gray, self.prev_keypoints, None)
        
        prev_keypoints = self.prev_keypoints[status == 1]
        matched_keypoints = matched_keypoints[status == 1]
        
        if len(matched_keypoints) == 0:
            matched_keypoints = np.empty((0, 2))
        
        if (np.size(prev_keypoints, 0) > 4) and (np.size(prev_keypoints, 0) == np.size(prev_keypoints, 0)):
            H, inliesrs = cv2.estimateAffinePartial2D(prev_keypoints, matched_keypoints, cv2.RANSAC)

            # Handle downscale
            if self.downscale_factor > 1.0:
                H[0, 2] *= self.downscale_factor
                H[1, 2] *= self.downscale_factor
        else:
            print('[LOGS] Not enough points for optical flow')
            
        self.prev_frame = frame_gray.copy()
        self.prev_keypoints = copy.copy(keypoints)
        
        return H
        
            
