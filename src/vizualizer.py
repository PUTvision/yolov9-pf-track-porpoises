from typing import List

import numpy as np
import cv2

from src.detection_result import DetectionResult


class Vizualizer:
    def __init__(self) -> None:
        pass

    def draw_tracks(self, frame: np.ndarray, track_predictions: np.ndarray) -> np.ndarray:
        for track in track_predictions:
            x1, y1, x2, y2 = track.xyxy
            color = track.color
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{track.track_id} {track.confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        

        return frame

    def draw_predictions(self, frame: np.ndarray, predictions: List[DetectionResult]) -> np.ndarray:
        
        for pred in predictions:
            x1, y1, x2, y2 = pred.xyxy
            color = pred.color
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{pred.label} {pred.confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame
