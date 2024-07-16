import os
import cv2
from pathlib import Path



class SaveResults:
    def __init__(self, root, sequence, save_rois=False, disable_keypoints=False):
        self.save_rois = save_rois
        self.disable_keypoints = disable_keypoints
        
        Path(f'{root}/data/').mkdir(parents=True, exist_ok=True)
        self.file = open(f'{root}/data/{sequence}.txt', 'w')
        
        if not self.disable_keypoints:
            self.keypoints_file = open(f'{root}/data/{sequence}_keypoints.txt', 'w')
        
        if self.save_rois:
            self.rois_dir = Path(f'{root}/{sequence}_rois')
            self.rois_dir.mkdir(parents=True, exist_ok=True)
            self.rois_padding = 10

    def update(self, frame_id, frame, track_predictions):
        frame_id = int(frame_id) + 1
        
        for track in track_predictions:
            x1, y1, x2, y2 = track.xyxy
            conf = track.confidence
            
            w = x2 - x1
            h = y2 - y1
            
            self.file.write(f'{frame_id},{int(track.track_id)},{x1},{y1},{w},{h},{conf},-1,-1,-1\n')
            
            if not self.disable_keypoints:
                tongue, tail = track.keypoints.get_tongue_tail()
                
                if tongue is not None:
                    tongue = (tongue[0] + x1, tongue[1] + y1)
                else:
                    tongue = (-1, -1)
                    
                if tail is not None:
                    tail = (tail[0] + x1, tail[1] + y1)
                else:
                    tail = (-1, -1)
                    
                if tongue != (-1, -1) and tail != (-1, -1):
                    self.keypoints_file.write(f'{frame_id},{int(track.track_id)},{tongue[0]},{tongue[1]},{tail[0]},{tail[1]}\n')
            
            if self.save_rois:
                Path(f'{self.rois_dir}/{track.track_id:03d}').mkdir(parents=True, exist_ok=True)
                
                # make a square roi
                # h = max(w, h)
                # w = h
                
                x1 = max(0, x1 - self.rois_padding)
                y1 = max(0, y1 - self.rois_padding)
                
                x2 = min(frame.shape[1], x2 + self.rois_padding)
                y2 = min(frame.shape[0], y2 + self.rois_padding)
                
                roi = frame[y1:y2, x1:x2]
                
                cv2.imwrite(f'{self.rois_dir}/{track.track_id:03d}/{frame_id:04d}.jpg', roi)
    
    def update_ocsort(self, frame_id, frame, ocsort_predictions):
        frame_id = int(frame_id) + 1
        
        for t in ocsort_predictions:            
            tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
            tid = int(t[4])
            
            if tid == 0:
                raise ValueError('Track ID cannot be 0')
            
            self.file.write(f'{frame_id},{int(tid)},{tlwh[0]},{tlwh[1]},{tlwh[2]},{tlwh[3]},1,-1,-1,-1\n')
    
    def update_botsort(self, frame_id, frame, botsort_predictions):
        frame_id = int(frame_id) + 1
        
        for t in botsort_predictions:
            tid = int(t.track_id)
            conf = t.score
            t, l, w, h = t.tlwh
            t = int(t)
            l = int(l)
            w = int(w)
            h = int(h)
            
            if tid == 0:
                raise ValueError('Track ID cannot be 0')
            
            self.file.write(f'{frame_id},{int(tid)},{t},{l},{w},{h},{conf},-1,-1,-1\n')
            
    def update_strongsort(self, frame_id, frame, tracks):
        frame_id = int(frame_id) + 1
        
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            bbox = track.to_tlwh()
            tid = track.track_id
            
            self.file.write(f'{frame_id},{int(tid)},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},1,-1,-1,-1\n')
    
    def save(self):
        self.file.close()
