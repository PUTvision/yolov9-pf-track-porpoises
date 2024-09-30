from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.track import Track, TrackParams
from src.detection_result import DetectionResult
from src.track_state import TrackState


class Sort:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.4, particle = False, flow = False) -> None:
        self.iou_threshold = iou_threshold
        self.use_flow = flow
        
        self.trackers = []
        self.tracks_counter = 0
        self.initialized = False
        self.prev_frame = None
                
        self.default_track_params = TrackParams(
            use_particles=particle,
            max_age=max_age,
            max_particle=max_age*2,
            min_hits=min_hits
        )
        
        if self.use_flow:
            from src.sparse_optical_flow import SparseOpticalFlow
            self.flow_estimator = SparseOpticalFlow()
    
    def __call__(self, frame, frame_index, predictions):
        return self._update(frame, frame_index, predictions)
    
    def _add(self, predictions: List[DetectionResult], state=TrackState.NEW):
        for pred in predictions:
            self.tracks_counter += 1
            self.trackers.append(Track(self.tracks_counter, pred, state, self.default_track_params))
    
    def _update(self, frame: np.ndarray, frame_index: int, predictions: List[DetectionResult]):        
        if self.prev_frame is None:
            self.prev_frame = frame
        
        if not self.initialized and len(predictions) > 0:
            self._add(predictions, state=TrackState.CONFIRMED)
            self.initialized = True
            self.prev_frame = frame
            return [track for track in self.trackers if track.is_confirmed]

        if self.use_flow:
            H = self.flow_estimator.update(frame, frame_index)
        
        for track in self.trackers:
            track.step(warp=H if self.use_flow else None)
            
            if track.is_dead:
                self.trackers.remove(track)
            
        matched, unmatched_dets, unmatched_tracks = self._associate_detections_to_trackers(frame, predictions, warp=H if self.use_flow else None)
        
        for track_idx, pred in matched:
            self.trackers[track_idx].update(pred)
            
        for track_idx in unmatched_tracks:
            self.trackers[track_idx].mark_missed()
            
        for detection_idx in unmatched_dets:
            self._add([predictions[detection_idx]])
        
        self.prev_frame = frame
        return [track for track in self.trackers if track.is_confirmed]
        
                
    def _associate_detections_to_trackers(self, frame, detections: List[DetectionResult], warp: Optional[np.ndarray] = None):
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
            
        if self.default_track_params.use_particles:
            matches, unmatched_detections, unmatched_trackers = \
                self.associate_using_particles(frame, detections, warp, matches, unmatched_detections, unmatched_trackers)
            
        return matches, unmatched_detections, unmatched_trackers
    
    def associate_using_particles(self, frame, detections, warp, matches, unmatched_detections, unmatched_trackers):
        [self.trackers[t].particle_step(frame, self.prev_frame, warp) for t in unmatched_trackers]
        
        if len(unmatched_detections) > 0:
            matches, unmatched_detections, unmatched_trackers = \
                self._associate_detections_using_particles(detections, matches, unmatched_detections, unmatched_trackers)
            
        if len(unmatched_trackers) > 0:
            matches, unmatched_trackers = \
                self._associate_tracks_using_particles(matches, unmatched_trackers)
            
        return matches, unmatched_detections, unmatched_trackers
        
    def _associate_detections_using_particles(self, detections, matches, unmatched_detections, unmatched_trackers):
        particle_threshold = 0.5 * self.iou_threshold
        
        particles_xyxys = np.array([self.trackers[t].particle_xyxy for t in unmatched_trackers if self.trackers[t].is_confirmed])
        detections_xyxys = np.array([detections[d_idx].xyxy for d_idx in unmatched_detections])
        
        if len(particles_xyxys) == 0:
            return matches, unmatched_detections, unmatched_trackers
        
        iou_matrix = self._iou_batch(particles_xyxys, detections_xyxys)
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > particle_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                y, x = linear_sum_assignment(-iou_matrix)
                matched_indices = np.array(list(zip(y, x)))
        else:
            matched_indices = np.empty(shape=(0,2))
            
        detections_ids_to_remove = []
        # handle matches
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] > particle_threshold:
                
                det = detections[unmatched_detections[m[1]]]
                det.particle = True
                
                matches.append((unmatched_trackers[m[0]], det))
                detections_ids_to_remove.append(unmatched_detections[m[1]])
                
        unmatched_detections = [d for d in unmatched_detections if d not in detections_ids_to_remove]
        
        return matches, unmatched_detections, unmatched_trackers
    
        
    def _associate_tracks_using_particles(self, matches, unmatched_trackers):
        
        particle_threshold = 0.5 * self.iou_threshold

        trackers_xyxys = np.array([self.trackers[t].xyxy for t in unmatched_trackers if self.trackers[t].is_confirmed])
        particles_xyxys = np.array([self.trackers[t].particle_xyxy for t in unmatched_trackers if self.trackers[t].is_confirmed])
        
        if len(particles_xyxys) == 0:
            return matches, unmatched_trackers
        
        iou_matrix = self._iou_batch(particles_xyxys, trackers_xyxys)
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > particle_threshold).astype(np.int32)
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
            if iou_matrix[m[0], m[1]] > particle_threshold:
                
                x1, y1, x2, y2 = particles_xyxys[m[0]]
                
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
        
        return matches, unmatched_trackers
    
    
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
