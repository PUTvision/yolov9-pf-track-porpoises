import os
from pathlib import Path



class SaveResults:
    def __init__(self, root, tracker, sequence):
        results_path = os.path.join(root, tracker)
        
        Path(f'{results_path}').mkdir(parents=True, exist_ok=True)
        self.file = open(f'{results_path}/{sequence}.txt', 'w')

    def update(self, frame_id, track_predictions):
        for track in track_predictions:
            x, y, w, h = track.xywh
            label = track.label
            conf = track.confidence
            
            frame_id = int(frame_id) + 1
            
            self.file.write(f'{frame_id} {int(track.track_id)} {x} {y} {w} {h} {label} {conf}\n')
                
    def save(self):
        self.file.close()
