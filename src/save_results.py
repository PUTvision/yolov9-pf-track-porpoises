import os
import cv2
from pathlib import Path



class SaveResults:
    def __init__(self, root, tracker, sequence, save_rois=False):
        results_path = os.path.join(root, tracker)
        self.save_rois = save_rois
        
        Path(f'{results_path}').mkdir(parents=True, exist_ok=True)
        self.file = open(f'{results_path}/{sequence}.txt', 'w')
        
        if self.save_rois:
            self.rois_dir = Path(f'{results_path}/{sequence}_rois')
            self.rois_dir.mkdir(parents=True, exist_ok=True)
            self.rois_padding = 10

    def update(self, frame_id, frame, track_predictions):
        frame_id = int(frame_id) + 1
        
        for track in track_predictions:
            x, y, w, h = track.xywh
            label = track.label
            conf = track.confidence
            
            self.file.write(f'{frame_id} {int(track.track_id)} {x} {y} {w} {h} {label} {conf}\n')
            
            if self.save_rois:
                Path(f'{self.rois_dir}/{track.track_id:03d}').mkdir(parents=True, exist_ok=True)
                
                # make a square roi
                # h = max(w, h)
                # w = h
                
                x1 = max(0, int(x - w//2) - self.rois_padding)
                y1 = max(0, int(y - h//2) - self.rois_padding)
                
                x2 = min(frame.shape[1], int(x + w//2) + self.rois_padding)
                y2 = min(frame.shape[0], int(y + h//2) + self.rois_padding)
                
                roi = frame[y1:y2, x1:x2]
                
                cv2.imwrite(f'{self.rois_dir}/{track.track_id:03d}/{frame_id:04d}.jpg', roi)
                
    def save(self):
        self.file.close()
