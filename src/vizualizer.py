from typing import List

import numpy as np
import cv2

from src.detection_result import DetectionResult


class Vizualizer:
    def __init__(self, out_name=None) -> None:
        self.out_name = out_name
        self.out = None
            
    def draw_tracks(self, frame: np.ndarray, track_predictions: np.ndarray) -> np.ndarray:
        if self.out_name is not None and self.out is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(f'{self.out_name}_out.avi', fourcc, 10.0, (frame.shape[1], frame.shape[0]))
        
        for track in track_predictions:
            x1, y1, x2, y2 = track.xyxy
            color = track.color
            
            if track.is_particle_active:
                for x, y in track.particle_particles:
                    cv2.circle(frame, (int(x), int(y)), 1, color, -1)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{track.track_id} {track.confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        if self.out is not None:
            self.out.write(frame)
        
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
