import click
import motmetrics as mm
import numpy as np
from glob import glob
import json
import os
from pathlib import Path


def xyxy2xywh(bbox: np.ndarray):
    return np.array([bbox[:, 0], bbox[:, 1], bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]]).T


@click.command()
@click.option('--results', '-r', help='Path to result txt file', required=True)
@click.option('--gt-path', '-g', help='Path to ground truth directory', default='./dataset/tracking/')
def main(results, gt_path):
    results = results.replace('//', '/')
    
    seq_name = results.split('/')[-1].split('.')[0]
    print(f'Evaluating {seq_name}')
    
    gt_file_path = os.path.join(gt_path, f'{seq_name}_tracking.txt')
    gt_info_path = os.path.join(gt_path, f'{seq_name}_info.json')
    
    pred = np.loadtxt(results, delimiter=' ').reshape(-1, 8)
    gt = np.loadtxt(gt_file_path, delimiter=' ').reshape(-1, 6)
    info = json.load(open(gt_info_path, 'r'))
    
    acc = mm.MOTAccumulator(auto_id=True)
    
    for frame_id in range(info['images']):
        gt_frame_id = frame_id
        pred_frame_id = frame_id + 1
        
        gt_dets = gt[np.int32(gt[:, 0]) == gt_frame_id, 1:]
        pred_dets = pred[np.int32(pred[:, 0]) == pred_frame_id, 1:]
        
        gt_dets_bboxes = xyxy2xywh(gt_dets[:, 1:5])
        pred_dets_bboxes = pred_dets[:, 1:5]
        
        C = mm.distances.iou_matrix(gt_dets_bboxes, pred_dets_bboxes)
    
        acc.update(gt_dets[:,0].astype('int').tolist(), pred_dets[:,0].astype('int').tolist(), C)
        
    mh = mm.metrics.create()
    
    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                     'recall', 'precision', 'num_objects', \
                                     'mostly_tracked', 'partially_tracked', \
                                     'mostly_lost', 'num_false_positives', \
                                     'num_misses', 'num_switches', \
                                     'num_fragmentations', 'mota', 'motp' \
                                    ],
                                    name='acc')    
    
    strsummary = mm.io.render_summary(
      summary,
      #formatters={'mota' : '{:.2%}'.format},
      namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
               'precision': 'Prcn', 'num_objects': 'GT', \
               'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
               'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
               'num_misses': 'FN', 'num_switches' : 'IDsw', \
               'num_fragmentations' : 'FM', 'mota': 'MOTA', 'motp' : 'MOTP',  \
              }
    )
    print(strsummary)
    
    algh_name = results.split('/')[-2]

    Path(f'./results/eval/{algh_name}').mkdir(parents=True, exist_ok=True)
    with open(f'./results/eval/{algh_name}/{seq_name}.json', 'w') as f:
        json.dump(summary.to_dict(), f)
    
    
if __name__ == '__main__':
    main()
