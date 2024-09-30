from typing import Optional, Tuple

import numpy as np

from filterpy.kalman import KalmanFilter

from src.detection_result import DetectionResult

# Modified from:
# https://github.com/RizwanMunawar/yolov7-object-tracking/blob/main/sort.py#L70
class KalmanBoxTracker(object):
    
    count = 0
    def __init__(self, det: DetectionResult):
        """
        Initialize a tracker using initial bounding box
        
        Parameter 'bbox' must have 'detected class' int number at the -1 position.
        """
        self.det = det
        
        bbox = det.xyxy
        
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10. # R: Covariance matrix of measurement noise (set to high for noisy inputs -> more 'inertia' of boxes')
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.5 # Q: Covariance matrix of process noise (set to high for erratically moving things)
        self.kf.Q[4:,4:] *= 0.5

        self.kf.x[:4] = self._convert_bbox_to_z(bbox) # STATE VECTOR
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.centroidarr = []
        CX = (bbox[0]+bbox[2])//2
        CY = (bbox[1]+bbox[3])//2
        self.centroidarr.append((CX,CY))
        
    def update(self, det: DetectionResult):
        """
        Updates the state vector with observed bbox
        """
        self.det = det
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        bbox = det.xyxy
        self.kf.update(self._convert_bbox_to_z(bbox))
        CX = (bbox[0]+bbox[2])//2
        CY = (bbox[1]+bbox[3])//2
        self.centroidarr.append((CX,CY))
    
    def predict(self, warp: Optional[np.ndarray] = None):
        """
        Advances the state vector and returns the predicted bounding box estimate
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        
        if warp is not None:
            self.apply_warp(warp)
   
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        
        x1, y1, x2, y2 = self.history[-1][0]
        
        return DetectionResult(
            label=self.det.label,
            confidence=self.det.confidence,
            x=(x1+x2)/2,
            y=(y1+y2)/2,
            w=x2-x1,
            h=y2-y1,
            color=self.det.color,
            particle=self.det.particle,
        )
    
    def apply_warp(self, warp: np.array):
        R = warp[:2,:2]
        T = warp[:2,2]/2
        
        xy = self.kf.x[:2].reshape((2,))
        xy = np.dot(R, xy) + T
        
        xy_corr = self.kf.P[:2,:2]
        xy_corr = np.dot(R, xy_corr)
        xy_corr = np.dot(xy_corr, R.T)
        
        self.kf.x[:2] = xy.reshape((2,1))
        self.kf.P[:2,:2] = xy_corr
    
    @staticmethod
    def _convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    
        #scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def _convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if(score==None):
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
