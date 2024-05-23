from typing import List, Optional

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from src.detection_result import DetectionResult

np.random.seed(0)


class PFBoxTracker(object):
    NUM_PARTICLES = 200
    VEL_RANGE = [0.5, 0.5]
    
    def __init__(self, frame: np.ndarray, dets: List[DetectionResult]):
        self.particles = None
        self.median_color = None
        self.img_h, self.img_w = None, None
        
        self._initialize_particles(frame, dets)
    
    def predict(self, frame: np.ndarray, warp: Optional[np.ndarray] = None):
        self._apply_velocity(warp)
        self._enforce_edges()
        self.errors = self._compute_erros(frame)
        
        weights = self._compute_weights(self.errors.copy())        
        self._resaple(weights)
        self._apply_noise()
        
        self._update_template(frame)
    
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
        
        if len(history) == 1:
            vel_x, vel_y = 0, 0
        else:
            vel_x = (history[-1].x - history[0].x) / (len(history) - 1)
            vel_y = (history[-1].y - history[0].y) / (len(history) - 1)
                    
        particles_xy = np.random.normal(0, 1, (self.NUM_PARTICLES, 2))
        particles_xy *= np.array((w//2, h//2))
        particles_xy += np.array((x, y))
        
        if len(history) == 1:
            particles_vel_xy = np.random.normal(0, 1, (self.NUM_PARTICLES, 2))
            particles_vel_xy *= np.array(self.VEL_RANGE)
            particles_vel_xy -= np.array(self.VEL_RANGE)/2.0
        else:
            particles_vel_xy = np.random.normal(0, 1, (self.NUM_PARTICLES, 2))
            particles_vel_xy += np.array((vel_x, vel_y))
        
        self.particles = np.concatenate([particles_xy, particles_vel_xy], axis=1)
                
        if len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.hist = self._calculate_lbp_histogram(self._crop_roi(frame, x, y, w, h))
    
    def _apply_velocity(self, warp: Optional[np.ndarray] = None):
        self.particles[:, 0] += self.particles[:, 2]  # x = x + u
        self.particles[:, 1] += self.particles[:, 3]    
        
    def _enforce_edges(self):
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.img_w - 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.img_h - 1)
        
    def _compute_erros(self, frame: np.ndarray):
        xs, ys = self.particles[:, 0], self.particles[:, 1]
        h, w = self.w, self.h
        
        if len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        errors = []        
        for x, y in zip(xs, ys):
            hist = self._calculate_lbp_histogram(self._crop_roi(frame, x, y, w, h))
            
            if self.hist.shape != hist.shape:
                if len(errors) == 0:
                    errors.append(1e6)
                else:
                    errors.append(np.max(errors))
                continue
    
            local_error = np.linalg.norm(self.hist - hist)
            
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
            
    def _resaple(self, weights: np.ndarray):
        probabilities = weights / (np.sum(weights))

        # Resample
        indices = np.random.choice(
            self.NUM_PARTICLES,
            size=self.NUM_PARTICLES,
            p=probabilities)
        
        self.particles = self.particles[ indices, : ]

    def _apply_noise(self):
        POS_SIGMA = 1.0
        VEL_SIGMA = 0.5
        
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

    def _update_template(self, frame: np.ndarray):
        xy_particle = self.get_center()
        x, y = int(xy_particle[0]), int(xy_particle[1])
        
        if len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        self.hist = self._calculate_lbp_histogram(self._crop_roi(frame, x, y, self.w, self.h))

    @staticmethod
    def _calculate_lbp_histogram_once(img, n_points=8, radius=1, histogram_size=11):
        lbp = local_binary_pattern(img, n_points, radius, 'uniform')
        
        hist, _ = np.histogram(lbp, bins=histogram_size, range=(0, histogram_size), density=True)
        
        return hist

    def _calculate_lbp_histogram(self, img):
        
        hist1 = self._calculate_lbp_histogram_once(img, 8, 1, 11)
        hist2 = self._calculate_lbp_histogram_once(img, 8, 2, 11)
        hist3 = self._calculate_lbp_histogram_once(img, 8, 3, 11)

        return np.concatenate([hist1, hist2, hist3])

    @staticmethod
    def _crop_roi(frame, x, y, w, h):
        x1o = max(x - w // 2 - 5, 0)
        y1o = max(y - h // 2 - 5, 0)
        x2o = min(x + w // 2 + 5, frame.shape[1])
        y2o = min(y + h // 2 + 5, frame.shape[0])
        
        x1o = int(x1o)
        y1o = int(y1o)
        x2o = int(x2o)
        y2o = int(y2o)
        
        roi = frame[y1o:y2o, x1o:x2o]

        return roi
