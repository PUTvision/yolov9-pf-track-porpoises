from typing import List

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from src.detection_result import DetectionResult


class PFBoxTracker(object):
    NUM_PARTICLES = 200
    VEL_RANGE = [0.5, 0.5]
    
    def __init__(self, frame: np.ndarray, dets: List[DetectionResult]):
        self.particles = None
        self.median_color = None
        self.img_h, self.img_w = None, None
        
        self._initialize_particles(frame, dets)
    
    def predict(self, frame: np.ndarray):                        
        self._apply_velocity()
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
        
        history = dets[-3:]
        
        if len(history) == 1:
            vel_x, vel_y = 0, 0
        elif len(history) == 2:
            vel_x = history[-1].x - history[-2].x
            vel_y = history[-1].y - history[-2].y
        else:
            vel_x = (history[-1].x - history[0].x) / (len(history) - 1)
            vel_y = (history[-1].y - history[0].y) / (len(history) - 1)
        
        self.particles = np.random.normal(0, 1, (self.NUM_PARTICLES, 4))
        self.particles = self.particles * np.array( (w//4, h//4, self.VEL_RANGE[0], self.VEL_RANGE[1]))
        self.particles = self.particles + np.array( (x, y, 0, 0))
        
        self.particles[:, 2:4] -= np.array(self.VEL_RANGE)/2.0 # Center velocities around 0
        self.particles[:, 2] += vel_x
        self.particles[:, 3] += vel_y
                
        if len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.hist = self._calculate_lbp_histogram(self._crop_roi(frame, x, y, w, h))
          
    def _apply_velocity(self):
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
    def _calculate_lbp_histogram(img, n_points=11, radius=3, method='uniform'):
        lbp = local_binary_pattern(img, n_points, radius, method)
        
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

        return hist

    @staticmethod
    def _crop_roi(frame, x, y, w, h):
        return frame[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
