import click
import cv2
from tqdm import tqdm

from src.frames_source import FramesSource
from src.sort import Sort
from src.save_results import SaveResults
from src.vizualizer import Vizualizer
from src.yolov9 import YOLOv9

@click.command()
@click.option('--task', '-t', help='Task to perform', required=True, type=click.Choice(['pred', 'viz', 'vid']))
@click.option('--tracker', '-tr', help='Tracker to use', default='none', type=click.Choice(['none', 'sort', 'sort-pf']))
@click.option('--source', '-s', help='Source of the video (video file or catalog)', required=True)
@click.option('--model-path', help='Path to the model weight', default='./data/best.onnx')
@click.option('--engine', help='Engine to use', default='cuda', type=click.Choice(['cuda', 'cpu']))
def main(task: str, tracker: str, source: str, model_path: str, engine: str):
    
    detector = YOLOv9(model_path, engine)
    frame_source = FramesSource(source)

    if tracker == 'sort' or tracker == 'sort-pf':
        track = Sort(particle='pf' in tracker)
        
    if task == 'viz':
        cv2.namedWindow('Video')
        vizualizer = Vizualizer()
    
    if task == 'vid':
        vizualizer = Vizualizer(
            out_name = source.split('/')[-2] if source.endswith('/') else source.split('/')[-1].split('.')[0],
        )
    
    if task == 'pred':
        save_results = SaveResults(
            root='./results/',
            tracker=tracker,
            sequence=source.split('/')[-2] if source.endswith('/') else source.split('/')[-1].split('.')[0] 
            )
        
    print(f'[LOGS] Start tracking...')
    
    for frame, index, frame_name in tqdm(frame_source):
        predictions = detector.predict(frame)
        
        if tracker != 'none':
            track_predictions = track(frame, index, predictions)
            
        if task == 'viz' or task == 'vid':
            if tracker != 'none':
                frame = vizualizer.draw_tracks(frame, track_predictions)
            else:
                frame = vizualizer.draw_predictions(frame, predictions)
                
        if task == 'pred':
            if tracker == 'none':
                raise NotImplementedError('Prediction task is not implemented for tracking')
            
            save_results.update(index, track_predictions)
            
        if task == 'viz':
            cv2.imshow('Video', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1)
            
            if key == 27:
                break
        
    if task == 'pred':
        save_results.save()
        
    
if __name__ == '__main__':
    main()
