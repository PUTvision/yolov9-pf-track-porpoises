import click
import cv2
from tqdm import tqdm

from src.frames_source import FramesSource
from src.sort import Sort
from src.save_results import SaveResults
from src.vizualizer import Vizualizer
from src.yolov9 import YOLOv9
from src.yolov7 import YOLOv7
from src.keypoints import KeyPoints


@click.command()
@click.option('--task', '-t', help='Task to perform', required=True, type=click.Choice(['pred', 'viz', 'vid', 'rois']))
@click.option('--tracker', '-tr', help='Tracker to use', default='none', type=click.Choice(['none', 'sort', 'sort-pf']))
@click.option('--source', '-s', help='Source of the video (video file or catalog)', required=True)
@click.option('--model', '-m', help='Model to use', default='yolov9', type=click.Choice(['yolov9', 'yolov7']))
@click.option('--model-path', help='Path to the model weight', default='./data/best-yolov9.onnx')
@click.option('--kmodel-path', help='Path to the keypoints model weight', default='./data/porpoises_keypoints_128_all.onnx')
@click.option('--engine', help='Engine to use', default='cuda', type=click.Choice(['cuda', 'cpu']))
@click.option('--disable-viz', help='Disable visualization', is_flag=True)
def main(task: str, tracker: str, source: str, model: str, model_path: str, kmodel_path: str, engine: str, disable_viz: bool):
    
    if model == 'yolov7':
        detector = YOLOv7(model_path, engine)
    elif model == 'yolov9':
        detector = YOLOv9(model_path, engine)
        
    keypoints = KeyPoints(kmodel_path, engine)
    frame_source = FramesSource(source)

    if tracker == 'sort' or tracker == 'sort-pf':
        track = Sort(particle='pf' in tracker)
        
    if task == 'viz':
        cv2.namedWindow('Video')
        vizualizer = Vizualizer(disable_viz=disable_viz)
    
    if task == 'vid':
        vizualizer = Vizualizer(
            out_name = source.split('/')[-2] if source.endswith('/') else source.split('/')[-1].split('.')[0],
            disable_viz = disable_viz,
        )
    
    if task == 'pred' or task == 'rois':
        save_results = SaveResults(
            root=f'./results/{model}/',
            tracker=tracker,
            sequence=source.split('/')[-2] if source.endswith('/') else source.split('/')[-1].split('.')[0],
            save_rois=task == 'rois',
            )
        
    print(f'[LOGS] Start tracking...')
    
    for frame, index, frame_name in tqdm(frame_source):
        predictions = detector.predict(frame)
        
        if tracker != 'none':
            track_predictions = track(frame, index, predictions)
            
            keypoints.predict_and_update(frame, track_predictions)
            
        if task == 'viz' or task == 'vid':
            if tracker != 'none':
                frame = vizualizer.draw_tracks(frame, track_predictions)
            else:
                frame = vizualizer.draw_predictions(frame, predictions)
                
        if task == 'pred' or task == 'rois':
            if tracker == 'none':
                raise NotImplementedError('Prediction task is not implemented for tracking')
            
            save_results.update(index, frame, track_predictions)
            
        if task == 'viz':
            cv2.imshow('Video', cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (1280, int(1280 * frame_source.height / frame_source.width))))
            key = cv2.waitKey(1)
            
            if key == 27:
                break
        
    if task == 'pred':
        save_results.save()
        
    
if __name__ == '__main__':
    main()
