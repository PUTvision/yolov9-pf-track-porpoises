from typing import List

import numpy as np
import cv2

from src.detection_result import DetectionResult


class Vizualizer:
    def __init__(self, out_name=None, disable_viz=False, disable_keypoints: bool = False) -> None:
        self.out_name = out_name
        self.out = None
        self.disable_viz = disable_viz
        self.disable_keypoints = disable_keypoints
            
    def draw_tracks(self, frame: np.ndarray, track_predictions: np.ndarray) -> np.ndarray:
        if self.out_name is not None and self.out is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(f'{self.out_name}_out.avi', fourcc, 10.0, (frame.shape[1], frame.shape[0]))
        
        for track in track_predictions:
            x1, y1, x2, y2 = track.xyxy
            color = track.color
            
            roi = cv2.cvtColor(frame[max(0, y1-5):min(y2+5, frame.shape[0]), max(x1-5, 0):min(x2+5, frame.shape[1])], cv2.COLOR_BGR2GRAY)
            gaussed_roi = cv2.GaussianBlur(roi, (7, 7), 0)
            
            diff = cv2.absdiff(roi, gaussed_roi)
            
            is_above = (np.mean(diff) > 3 and np.std(diff) > 6)
            
            if not self.disable_keypoints:
                tongue, tail = track.keypoints.get_tongue_tail()
                
                if tongue is not None:
                    tongue = (tongue[0] + x1, tongue[1] + y1)
                    cv2.circle(frame, tongue, 4, (255, 0, 255), 2) # purple (255, 0, 255)
                    
                if tail is not None:
                    tail = (tail[0] + x1, tail[1] + y1)
                    cv2.circle(frame, tail, 4, (0, 165, 255), 2) # orange (0, 165, 255)
            
            x1 = max(int(x1 - 10), 0)
            y1 = max(int(y1 - 10), 0)
            
            x2 = min(int(x2 + 10), frame.shape[1] - 1)
            y2 = min(int(y2 + 10), frame.shape[0] - 1)
            
            if track.is_particle_active and not self.disable_viz:
                for x, y in track.particle_particles:
                    cv2.circle(frame, (int(x), int(y)), 1, color, -1)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            if not self.disable_viz:
                cv2.putText(frame, f'{track.track_id} {track.confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
            # if is_above:
            #     cv2.rectangle(frame, (x1-10, y1-10), (x1, y1), (255, 255, 255), -1)
            # else:
            #     cv2.rectangle(frame, (x1-10, y1-10), (x1, y1), (0, 0, 0), -1)
            
        if self.out is not None:
            self.out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        return frame
    
    def draw_tracks_ocsort(self, frame: np.ndarray, ocsort_predictions: np.ndarray, frame_shape: tuple) -> np.ndarray:
        for t in ocsort_predictions:
            x1, y1, x2, y2 = map(int, t[:4])
            tid = int(t[4])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{tid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        return frame

    def draw_tracks_botsort(self, frame: np.ndarray, botsort_predictions: np.ndarray, frame_shape: tuple) -> np.ndarray:
        for t in botsort_predictions:
            tid = int(t.track_id)
            t, l, w, h = t.tlwh
            
            x1 = int(t)
            y1 = int(l)
            
            x2 = int(t + w)
            y2 = int(l + h)
            
            if tid == 0:
                raise ValueError('Track ID cannot be 0')          
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{tid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        return frame

    def draw_tracks_strongsort(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            bbox = track.to_tlwh()
            tid = track.track_id
            
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'{tid}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        return frame

    def draw_predictions(self, frame: np.ndarray, predictions: List[DetectionResult]) -> np.ndarray:
        
        for pred in predictions:
            x1, y1, x2, y2 = pred.xyxy
            color = pred.color
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{pred.label} {pred.confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame

    def __del__(self):
        if self.out is not None:
            self.out.release()
            
        cv2.destroyAllWindows()
