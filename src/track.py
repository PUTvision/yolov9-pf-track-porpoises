from typing import Optional
from dataclasses import dataclass

import numpy as np
import random

from src.track_state import TrackState
from src.kalman_box_tracker import KalmanBoxTracker
from src.particles_filter_box_tracker import PFBoxTracker
from src.detection_result import DetectionResult
from src.track_state import TrackState

class ParticleWrapper:
    initialized = False

    def __init__(self):
        self._pf = None
        self._pos = None
        
    def predict(self, frame, prev_frame, preds, warp=None):
        if not self.initialized:
            self._pf = PFBoxTracker(prev_frame, preds)
            self.initialized = True
        
        self._pf._apply_velocity(preds)
        self._pf.predict(frame, warp)
        
    def deactivate(self):
        self.initialized = False
        self._pf = None
        self._pos = None

    def get_center(self):
        return self._pf.get_center()
    
    def get_particles(self):
        return self._pf.get_particles()
    
@dataclass
class TrackParams:
    use_particles: bool = False
    max_age: int = 5
    max_particle: int = 10,
    min_hits: int = 3
    
    
class Track:
    def __init__(self, track_id: int, pred: DetectionResult, state: TrackState, track_params: TrackParams) -> None:
        self._track_id = track_id
        self._pred = [pred]
        self._state = state
        self._track_params = track_params
        
        self._active_counter = 1
        self._missing_counter = 0
        self._particle_counter = 0
        
        self._kbt = KalmanBoxTracker(pred)
        self._pfbt = ParticleWrapper()
        self._pos = pred
        
        self._color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def step(self, warp: Optional[np.ndarray] = None):
        self._pos = self._kbt.predict(warp)
        self._limit_pred_history()
 
    
    def update(self, pred: DetectionResult):
        self._pred.append(pred)
        self._active_counter += 1
        self._missing_counter = 0
        
        self._kbt.update(pred)
            
        if self._track_params.use_particles and not pred.particle and self._pfbt.initialized:
            self._pfbt.deactivate()
            self._particle_counter = 0
        
        if self._track_params.use_particles and pred.particle:
            self._particle_counter += 1
            
        if self._active_counter > self._track_params.min_hits:
            self._state = TrackState.CONFIRMED
            
        if self._particle_counter > self._track_params.max_particle:
            self._pfbt.deactivate()
            self._particle_counter = 0
            self._state = TrackState.DEAD
        
    def mark_missed(self):
        self._state = TrackState.MISSING
        self._missing_counter += 1
        self._active_counter = 0
        
        if self._missing_counter > self._track_params.max_age:
            self._state = TrackState.DEAD

    @property
    def state(self):
        return self._state

    @property
    def is_confirmed(self):
        return self._state.is_confirmed()
    
    @property
    def is_dead(self):
        return self._state.is_dead()

    @property
    def xywh(self):
        x, y, w, h = self._pos.xywh
        return int(x), int(y), int(w), int(h)
    
    @property
    def xyxy(self):
        x1, y1, x2, y2 = self._pos.xyxy
        return int(x1), int(y1), int(x2), int(y2)

    @property
    def confidence(self):
        return self._pred[-1].confidence
    
    @property
    def color(self):
        return self._color
    
    @property
    def track_id(self):
        return self._track_id

    @property
    def label(self):
        return self._pred[-1].label

    @property
    def is_particle_active(self):
        return self._track_params.use_particles and self._pfbt.initialized
    
    def particle_step(self, frame: np.ndarray, prev_frame: np.ndarray, warp: Optional[np.ndarray] = None):
        self._pfbt.predict(frame, prev_frame, self._pred, warp)
    
    @property
    def particle_center(self):
        return self._pfbt.get_center()
    
    @property
    def particle_particles(self):
        return self._pfbt.get_particles()

    @property
    def particle_xyxy(self):
        if len(self._pred) < 2:
            _, _, w, h = self._pred[-1].xywh
        else:
            _, _, w, h = self._pred[-2].xywh
        xc, yc = self.particle_center
        
        new_x1 = int(xc - w / 2)
        new_y1 = int(yc - h / 2)
        new_x2 = int(xc + w / 2)
        new_y2 = int(yc + h / 2)
        
        return new_x1, new_y1, new_x2, new_y2

    def _limit_pred_history(self):
        self._pred = self._pred[-10:]

    