import random

from src.track_state import TrackState
from src.kalman_box_tracker import KalmanBoxTracker


class Track:
    def __init__(self, track_id, pred, state) -> None:
        self._track_id = track_id
        self._pred = [pred]
        self._state = state
        
        self._active_counter = 1
        self._missing_counter = 0
        
        self._kbt = KalmanBoxTracker(pred.xyxy)
        self._pos = pred.xyxy
        self._color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def step(self):
        self._pos = self._kbt.predict()[0]
        self._limit_pred_history()    
    
    def update(self, frame, pred):
        self._pred.append(pred)
        self._active_counter += 1
        self._missing_counter = 0
        
        self._kbt.update(pred.xyxy)
        
        if self._active_counter > 3:
            self._state = TrackState.CONFIRMED
        
    def mark_missed(self):
        self._state = TrackState.MISSING
        self._missing_counter += 1
        self._active_counter = 0
        
        if self._missing_counter > 5:
            self._state = TrackState.DEAD

    @property
    def state(self):
        return self._state

    @property
    def xywh(self):
        x1, y1, x2, y2 = self._pos
        xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        return int(xc), int(yc), int(w), int(h)
    
    @property
    def xyxy(self):
        x1, y1, x2, y2 = self._pos
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

    def _limit_pred_history(self):
        self._pred = self._pred[-10:]
