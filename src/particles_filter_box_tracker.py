from typing import List, Optional

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import joblib
from sklearn.ensemble import RandomForestClassifier


from src.detection_result import DetectionResult


np.random.seed(0)


class PFBoxTracker(object):
    NUM_PARTICLES = 200
    VEL_RANGE = [0.5, 0.5]
    
    def __init__(self, frame: np.ndarray, dets: List[DetectionResult]):
        self.particles = None
        self.median_color = None
        self.img_h, self.img_w = None, None
        self.hist = None
        
        self._initialize_particles(frame, dets)
        self._regressor: RandomForestClassifier = joblib.load('./data/regressor.joblib')
            
    def predict(self, frame: np.ndarray, warp: Optional[np.ndarray] = None):
        self._apply_velocity()
        self._enforce_edges()
        self.errors = self._compute_errors(frame)
        
        weights = self._compute_weights(self.errors.copy())        
        self._resample(weights)
        self._apply_noise()
    
    def get_center(self):
        return np.mean(self.particles[:, 0]), np.mean(self.particles[:, 1])
    
    def get_particles(self):
        return self.particles[:, 0:2]
    
    def _initialize_particles(self, frame: np.ndarray, dets: List[DetectionResult]):
        x, y, w, h = dets[-1].xywh
        
        self.w, self.h = w, h
        
        if self.img_h is None:
            self.img_h, self.img_w = frame.shape[:2]
        
        history = dets[-5:]
        random_vel = False
        
        if len(history) == 1:
            random_vel = True
        else:
            vel_x = (history[-1].x - history[0].x) / (len(history) - 1)
            vel_y = (history[-1].y - history[0].y) / (len(history) - 1)
            
            if np.abs(vel_x) < 2 and np.abs(vel_y) < 2:
                random_vel = True
                    
        particles_xy = np.random.normal(0, 1, (self.NUM_PARTICLES, 2))
        particles_xy *= np.array((w//2, h//2))
        particles_xy += np.array((x, y))
        
        if random_vel:
            particles_vel_xy = np.random.normal(0, 1, (self.NUM_PARTICLES, 2))
            particles_vel_xy *= np.array(self.VEL_RANGE)
            particles_vel_xy -= np.array(self.VEL_RANGE)/2.0
        else:
            particles_vel_xy = np.random.normal(0, 1, (self.NUM_PARTICLES, 2))
            particles_vel_xy += np.array((vel_x, vel_y))
        
        self.particles = np.concatenate([particles_xy, particles_vel_xy], axis=1)
        self.hist = self._calculate_lbp_histogram(self._crop_roi(frame, x, y, w, h))
    
    def _apply_velocity(self):
        self.particles[:, 0] += self.particles[:, 2]  # x = x + u
        self.particles[:, 1] += self.particles[:, 3]    
        
    def _enforce_edges(self):
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.img_w - 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.img_h - 1)
        
    def _compute_errors(self, frame: np.ndarray):
        xs, ys = self.particles[:, 0], self.particles[:, 1]
        h, w = self.w, self.h
        
        errors = []        
        for x, y in zip(xs, ys):
            hist = self._calculate_lbp_histogram(self._crop_roi(frame, x, y, w, h))
            
            if self.hist.shape != hist.shape:
                errors.append(1.0)
                continue
    
            # local_error = np.linalg.norm(self.hist - hist)     
            local_error = self._regressor.predict_proba((self.hist - hist).reshape(1, -1))[0][0]       
            errors.append(local_error)
        
        return np.array(errors)

    def _compute_weights(self, errors):
        weights = np.max(errors) - errors + 1e-6
        weights[ 
            (self.particles[ :,0 ] == 0) |
            (self.particles[ :,0 ] == self.img_w-1) |
            (self.particles[ :,1 ] == 0) |
            (self.particles[ :,1 ] == self.img_h-1)
        ] = 1e-6
        weights[np.isnan(weights)] = 1e-6
        
        return weights
            
    def _resample(self, weights: np.ndarray):
        probabilities = weights / (np.sum(weights))

        # Resample
        indices = np.random.choice(
            self.NUM_PARTICLES,
            size=self.NUM_PARTICLES,
            p=probabilities)
        
        self.particles = self.particles[ indices, : ]

    def _apply_noise(self):
        POS_SIGMA = 1.5
        VEL_SIGMA = 1.0
        
        noise = np.concatenate(
            (
                np.random.normal(0.0, POS_SIGMA, (self.NUM_PARTICLES,1)),
                np.random.normal(0.0, POS_SIGMA, (self.NUM_PARTICLES,1)),
                np.random.normal(0.0, VEL_SIGMA, (self.NUM_PARTICLES,1)),
                np.random.normal(0.0, VEL_SIGMA, (self.NUM_PARTICLES,1))
            ),
            axis=1
        )
        
        self.particles += noise

    def _update_template(self, frame: np.ndarray, pred: DetectionResult):
        x, y, w, h = pred.xywh
        
        new_hist = self._calculate_lbp_histogram(self._crop_roi(frame, x, y, self.w, self.h))
        self.hist = np.mean([self.hist, new_hist], axis=0)

    @staticmethod
    def _calculate_lbp_histogram_once(img, n_points=11, radius=3, method='uniform', histogram_size = None):
        lbp = local_binary_pattern(img, n_points, radius, method)
        
        if histogram_size is None:
            n_bins = int(lbp.max() + 1)
        else:
            n_bins = histogram_size
            
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

        return hist

    def _calculate_lbp_histogram(self, crop):
        "Img is in RGB format"
        
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        crop_h = crop_hsv[:,:,0]
    
        meadian = np.median(np.concatenate([crop_h[:,0], crop_h[:,-1], crop_h[0,:], crop_h[-1,:]], axis=0)).astype(np.uint8)
        
        crop_h_norm = crop_h - meadian
        crop_h_norm[crop_h_norm<0] = 0
        
        hist1 = self._calculate_lbp_histogram_once(crop_h_norm, n_points=11, radius=1, histogram_size=12)
        hist2 = self._calculate_lbp_histogram_once(crop_h_norm, n_points=11, radius=3, histogram_size=12)
        hist3 = self._calculate_lbp_histogram_once(crop_h_norm, n_points=11, radius=5, histogram_size=12)
    
        concat = np.concatenate([hist1, hist2, hist3])
    
        return concat

    @staticmethod
    def _crop_roi(frame, x, y, w, h):
        x1o = max(x - w // 2 - 10, 0)
        y1o = max(y - h // 2 - 10, 0)
        x2o = min(x + w // 2 + 10, frame.shape[1])
        y2o = min(y + h // 2 + 10, frame.shape[0])
        
        x1o = int(x1o)
        y1o = int(y1o)
        x2o = int(x2o)
        y2o = int(y2o)
        
        roi = frame[y1o:y2o, x1o:x2o]

        return roi
