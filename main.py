import click
import cv2
from tqdm import tqdm
import os, sys

from src.frames_source import FramesSource
from src.sort import Sort
from src.save_results import SaveResults
from src.vizualizer import Vizualizer
from src.yolov9 import YOLOv9
from src.keypoints import KeyPoints


def is_our_tracker(tracker: str) -> bool:
    return tracker in [
        'sort', 'sort-flow',
        'sort-pf', 'sort-pf-flow',
    ]

@click.command()
@click.option('--task', '-t', help='Task to perform', required=True, type=click.Choice(['pred', 'viz', 'vid', 'rois']))
@click.option('--tracker', '-tr', help='Tracker to use', default='none', type=str)
@click.option('--source', '-s', help='Source of the video (video file or catalog)', required=True)
@click.option('--weights', help='Path to the model weight', default='./data/best-yolov9.onnx')
@click.option('--cache-yolo', help='Cache YOLO predictions', is_flag=True)
@click.option('--kmodel-path', help='Path to the keypoints model weight', default='./data/porpoises_keypoints_128.onnx')
@click.option('--engine', help='Engine to use', default='cuda', type=click.Choice(['cuda', 'cpu']))
@click.option('--disable-viz', help='Disable additional visualization (only bounding boxes)', is_flag=True)
@click.option('--disable-particles', help='Disable particles filter visualization', is_flag=True)
@click.option('--enable-keypoints', help='Enable keypoints for porpoise (tongue, tail)', is_flag=True)
@click.option('--keypoint-thresh', help='Keypoint threshold', default=0.5)
def main(task: str, tracker: str, source: str, weights: str, cache_yolo: bool, kmodel_path: str, engine: str, disable_viz: bool, disable_particles: bool, enable_keypoints: bool, keypoint_thresh: float):
    detector = YOLOv9(weights, engine, cache_yolo=cache_yolo,
                        video_name=source.split('/')[-2] if source.endswith('/') else source.split('/')[-1].split('.')[0])
    
    if enable_keypoints:
        keypoints = KeyPoints(kmodel_path, engine, keypoint_thresh)
    
    frame_source = FramesSource(source)

    if is_our_tracker(tracker):
        track = Sort(
            particle='pf' in tracker,
            flow='flow' in tracker,
            )
        
    elif tracker == 'ocsort':
        sys.path.append('./trackers/OC_SORT/')
        from trackers.ocsort_tracker.ocsort import OCSort
        from src.ocsort_utils import preds_to_ocsort
        
        # default parameters
        ocsort_track = OCSort(
            det_thresh = 0.6, 
            iou_threshold= 0.3,
            asso_func="iou", 
            delta_t=3, 
            inertia=0.2, 
            use_byte=False,
        )
    elif 'botsort' in tracker:
        sys.path.append('./trackers/BoT-SORT/')
        from tracker.bot_sort import BoTSORT
        from src.botsort_utils import preds_to_botsort

        if '-' not in tracker:
            cmc_method = "none"
        else:
            cmc_method = tracker.split('-')[1]
            
        class BotArgs:
            track_high_thresh = 0.6
            track_low_thresh = 0.1
            new_track_thresh = 0.7
            track_buffer = 30
            match_thresh = 0.8
            aspect_ratio_thresh = 5
            min_box_area = 10
            fuse_score = False
            cmc_method = "none"
            proximity_thresh = 0.5
            appearance_thresh = 0.25
            with_reid = False
            name = "bot"
            ablation = False
            mot20 = False
        
        args = BotArgs()
        args.cmc_method = cmc_method
        
        botsort_tracker = BoTSORT(args, frame_rate=4)
    elif tracker == 'strongsort':
        sys.path.append('./trackers/StrongSORT/')
        from deep_sort.tracker import Tracker
        from deep_sort import nn_matching
        from src.strongsort_utils import preds_to_strongsort
        
        metric = nn_matching.NearestNeighborDistanceMetric(
            'cosine',
            0.2,
            None
        )
        
        strongsort_tracker = Tracker(
            metric
        )
        
    if task == 'viz':
        cv2.namedWindow('Video')
        vizualizer = Vizualizer(
            disable_viz=disable_viz, 
            disable_keypoints=not enable_keypoints, 
            disable_particles=disable_particles,
            )
    
    if task == 'vid':
        vizualizer = Vizualizer(
            out_name = source.split('/')[-2] if source.endswith('/') else source.split('/')[-1].split('.')[0],
            disable_viz = disable_viz,
            disable_keypoints=not enable_keypoints,
            disable_particles=disable_particles
        )
    
    if task == 'pred' or task == 'rois':
        save_results = SaveResults(
            root=f'./track_data/trackers/MOT17-test/{tracker}/',
            sequence=source.split('/')[-2] if source.endswith('/') else source.split('/')[-1].split('.')[0],
            save_rois=task == 'rois',
            disable_keypoints=not enable_keypoints,
            )
        
    print(f'[LOGS] Start tracking...')
    
    for frame, index, frame_name in tqdm(frame_source):
        predictions = detector.predict(frame, frame_id=index)
        
        if is_our_tracker(tracker):
            track_predictions = track(frame, index, predictions)
            
            if enable_keypoints:
                keypoints.predict_and_update(frame, track_predictions)
            
        elif tracker == 'ocsort':
            ocsort_predictions = ocsort_track.update(preds_to_ocsort(predictions), frame.shape[:2], frame.shape[:2])
        elif 'botsort' in tracker:
            botsort_preditions = botsort_tracker.update(preds_to_botsort(predictions), frame)
        elif tracker == 'strongsort':
            strongsort_tracker.predict()
            strongsort_tracker.update(preds_to_strongsort(predictions))
            
        if task == 'viz' or task == 'vid':
            if is_our_tracker(tracker):
                frame = vizualizer.draw_tracks(frame, track_predictions)
            elif tracker == 'ocsort':
                frame = vizualizer.draw_tracks_ocsort(frame, ocsort_predictions, frame.shape[:2])
            elif 'botsort' in tracker:
                frame = vizualizer.draw_tracks_botsort(frame, botsort_preditions, frame.shape[:2])
            elif tracker == 'strongsort':
                frame = vizualizer.draw_tracks_strongsort(frame, strongsort_tracker.tracks)
            else:
                frame = vizualizer.draw_predictions(frame, predictions)
                
        if task == 'pred' or task == 'rois':
            if tracker == 'none':
                raise NotImplementedError('Prediction task is not implemented for tracking')
            elif is_our_tracker(tracker):
                save_results.update(index, frame, track_predictions)
            elif tracker == 'ocsort':
                save_results.update_ocsort(index, frame, ocsort_predictions)
            elif 'botsort' in tracker:
                save_results.update_botsort(index, frame, botsort_preditions)
            elif tracker == 'strongsort':
                save_results.update_strongsort(index, frame, strongsort_tracker.tracks)
            
        if task == 'viz':
            cv2.imshow('Video', cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (1280, int(1280 * frame_source.height / frame_source.width))))
            key = cv2.waitKey(100 if cache_yolo else 1)
            
            if key == 27:
                break
        
    if task == 'pred':
        save_results.save()
        
    
if __name__ == '__main__':
    main()
