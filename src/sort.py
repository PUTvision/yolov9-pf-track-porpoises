from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.track import Track
from src.detection_result import DetectionResult
from src.track_state import TrackState


class Sort:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.4, particle = False) -> None:
        self.max = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.particle = particle
        
        self.trackers = []
        self.tracks_counter = 0
        self.initialized = False
    
    def __call__(self, frame, frame_index, predictions):
        return self._update(frame, frame_index, predictions)
    
    def _add(self, predictions: List[DetectionResult], state=TrackState.NEW):
        for pred in predictions:
            self.tracks_counter += 1
            self.trackers.append(Track(self.tracks_counter, pred, state, self.particle))
    
    def _update(self, frame: np.ndarray, frame_index: int, predictions: List[DetectionResult]):
        if not self.initialized and len(predictions) > 0:
            self._add(predictions, state=TrackState.CONFIRMED)
            self.initialized = True
            return [track for track in self.trackers if track.state.is_confirmed()]
        
        for track in self.trackers:
            track.step(frame)
            
            if track.state.is_dead():
                self.trackers.remove(track)
            
        matched, unmatched_dets, unmatched_tracks = self._associate_detections_to_trackers(frame, predictions)
        
        for track_idx, pred in matched:
            self.trackers[track_idx].update(pred)
            
        for track_idx in unmatched_tracks:
            self.trackers[track_idx].mark_missed(frame)
            
        for detection_idx in unmatched_dets:
            self._add([predictions[detection_idx]])
        
        return [track for track in self.trackers if track.is_confirmed]
        
                
    def _associate_detections_to_trackers(self, frame, detections: List[DetectionResult]):
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []
        
        unmatched_detections = []
        unmatched_trackers = []
        matches = []
        
        if len(detections) > 0:
            detections_xyxys = np.array([d.xyxy for d in detections])
            trackers_xyxys = np.array([t.xyxy for t in self.trackers])
            
            iou_matrix = self._iou_batch(detections_xyxys, trackers_xyxys)
            if min(iou_matrix.shape) > 0:
                a = (iou_matrix > self.iou_threshold).astype(np.int32)
                if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                    matched_indices = np.stack(np.where(a), axis=1)
                else:
                    y, x = linear_sum_assignment(-iou_matrix)
                    matched_indices = np.array(list(zip(y, x)))
            else:
                matched_indices = np.empty(shape=(0,2))
                
            # handle unmatched detections
            for d in range(len(detections)):
                if d not in matched_indices[:, 0]:
                    unmatched_detections.append(d)
                    
            # handle unmatched trackers
            for t in range(len(self.trackers)):
                if t not in matched_indices[:, 1]:
                    unmatched_trackers.append(t)
                    
            # handle matches
            for m in matched_indices:
                if iou_matrix[m[0], m[1]] < self.iou_threshold:
                    unmatched_detections.append(m[0])
                    unmatched_trackers.append(m[1])
                else:
                    matches.append((m[1], detections[m[0]]))
        else:
            unmatched_trackers = list(range(len(self.trackers)))
        
        if len(unmatched_trackers) > 0 and self.particle:
            _ = [self.trackers[t].particle_step(frame) for t in unmatched_trackers]
            
            trackers_xyxys = np.array([self.trackers[t].xyxy for t in unmatched_trackers])
            detections_particles_xyxys = np.array([self.trackers[t].particle_xyxy for t in unmatched_trackers])
            confirmed_trackers_ids = [i for i, t in enumerate(unmatched_trackers) if self.trackers[t].is_confirmed]
            
            iou_matrix = self._iou_batch(detections_particles_xyxys, trackers_xyxys)
            if min(iou_matrix.shape) > 0:
                a = (iou_matrix > self.iou_threshold).astype(np.int32)
                if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                    matched_indices = np.stack(np.where(a), axis=1)
                else:
                    y, x = linear_sum_assignment(-iou_matrix)
                    matched_indices = np.array(list(zip(y, x)))
            else:
                matched_indices = np.empty(shape=(0,2))
            
            tracks_ids_to_remove = []
            # handle matches
            for m in matched_indices:
                if iou_matrix[m[0], m[1]] > self.iou_threshold and m[0] in confirmed_trackers_ids:
                    
                    x1, y1, x2, y2 = detections_particles_xyxys[m[0]]
                    
                    new_det = DetectionResult(
                        label=self.trackers[unmatched_trackers[m[1]]].label,
                        confidence=self.trackers[unmatched_trackers[m[1]]].confidence,
                        x=(x1 + x2) // 2,
                        y=(y1 + y2) // 2,
                        w=x2 - x1,
                        h=y2 - y1,
                        particle=True
                    )
                    
                    matches.append((unmatched_trackers[m[1]], new_det))
                    tracks_ids_to_remove.append(unmatched_trackers[m[1]])
                    
            unmatched_trackers = [t for t in unmatched_trackers if t not in tracks_ids_to_remove]                  
        
        return matches, unmatched_detections, unmatched_trackers
    
    
    # https://github.com/RizwanMunawar/yolov7-object-tracking/blob/main/sort.py#L29C1-L44C14
    @staticmethod
    def _iou_batch(bb_test, bb_gt):        
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
        
        xx1 = np.maximum(bb_test[...,0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        return(o)
