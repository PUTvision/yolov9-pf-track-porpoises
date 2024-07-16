import cv2
import json
import os
from pathlib import Path


class FramesSource:
    def __init__(self, source: str):
        self.source = source
        
        self.counter = -1
        
        # Check if the source is a video file or a catalog
        if source.lower().endswith('.mp4') or source.lower().endswith('.avi') or source.lower().endswith('.mov'):
            self.source_type = 'video'
            self._video_name = source.split('/')[-1].split('.')[0]
        else:
            self.source_type = 'catalog'
        
        self.cap = self._get_cap(source)
        self.sensor_data = self._get_sensor_data(source)
        self.height = self._get_height()
        self.width = self._get_width()
        self.fps = self._get_fps()
        self.sensor_data_available = len(self.sensor_data) == len(self)
        
        print(f'[LOGS] Source type: {self.source_type} | Name: {self.source.split("/")[-1]} | Frames: {len(self)} | Resolution: {self.width}x{self.height} | FPS: {self.fps} | Sensor data: {self.sensor_data_available}')

    def index(self):
        return self.counter
    
    def _get_cap(self, source: str):
        if self.source_type == 'video':
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                raise ValueError(f'Error opening video source: {source}')
                
        else:
            cap = sorted([f'{source}/{f}' for f in os.listdir(source) if f.endswith('.jpg') or f.endswith('.png')])
            
            if len(cap) == 0:
                raise ValueError(f'No images found in the source: {source}')
            
        return cap
    
    def _get_sensor_data(self, source: str):
        if self.source_type == 'video':
            video_name = source.split('/')[-1].split('.')[0]
            catalog_dir = Path(source).parent.parent / 'sensors_data' / video_name
            
        else:
            catalog_name = source.split('/')[-2] if source.endswith('/') else source.split('/')[-1]
            catalog_dir = Path(source).parent.parent / 'sensors_data' / catalog_name
            
        if catalog_dir.exists():
            sensor_data = sorted([f'{catalog_dir}/{f}' for f in os.listdir(catalog_dir) if f.endswith('.json')])
        else:
            sensor_data = []
            
        return sensor_data
    
    def _get_height(self):
        if self.source_type == 'video':
            return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            return cv2.imread(self.cap[0]).shape[0]
        
    def _get_width(self):
        if self.source_type == 'video':
            return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        else:
            return cv2.imread(self.cap[0]).shape[1]
        
    def _get_fps(self):
        if self.source_type == 'video':
            return int(self.cap.get(cv2.CAP_PROP_FPS))
        else:
            return 1
        
    # Create generator to iterate over the frames using for loop
    def __len__(self):
        if self.source_type == 'video':
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            return len(self.cap)
        
    def __iter__(self):
        return self
    
    def __next__(self):
        self.counter += 1
        
        if self.source_type == 'video':
            frame_name = f'{self._video_name}_{self.counter}'
            ret, frame = self.cap.read()
            
            if not ret:
                raise StopIteration
            
        else:
            frame_name = self.cap[self.counter]
            frame = cv2.imread(frame_name)
            
            if self.counter == len(self.cap):
                raise StopIteration
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        sensor_data = {}
        
        if self.sensor_data_available:
            sensor_data = json.load(open(self.sensor_data[self.counter]))
        
        return frame, self.counter, frame_name, sensor_data
        
