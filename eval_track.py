import sys
import os
import argparse
from multiprocessing import freeze_support

sys.path.insert(0, 'TrackEval/')
import trackeval  # noqa: E402


freeze_support()

# Command line interface:
default_eval_config = trackeval.Evaluator.get_default_eval_config()
default_eval_config['DISPLAY_LESS_PROGRESS'] = False
default_eval_config['PRINT_RESULTS'] = False
default_eval_config['OUTPUT_SUMMARY'] = False
default_eval_config['OUTPUT_EMPTY_CLASSES'] = False
default_eval_config['OUTPUT_DETAILED'] = False
default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.0}
config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs


eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

# dataset_config['CLASSES_TO_EVAL'] = ['porpoise']
dataset_config['SPLIT_TO_EVAL'] = 'test'
dataset_config['DO_PREPROC'] = False
dataset_config['GT_FOLDER'] = './track_data/gt/'
dataset_config['TRACKERS_FOLDER'] = './track_data/trackers/'
dataset_config['SEQMAP_FILE'] = './track_data/testlist.txt'

# Run code
evaluator = trackeval.Evaluator(eval_config)
dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]

metrics_list = []
for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
    if metric.get_name() in metrics_config['METRICS']:
        metrics_list.append(metric(metrics_config))
if len(metrics_list) == 0:
    raise Exception('No metrics selected for evaluation')

x = evaluator.evaluate(dataset_list, metrics_list)

for alg in ['yolov9_sort', 'yolov9_sort-pf']:
    sequences = x[0]['MotChallenge2DBox'][alg].keys()
    
    metrics = {
        'HOTA': 0.0,
        'MOTA': 0.0,
        'MOTP': 0.0,
        'IDSW': 0.0,
        'IDF1': 0.0,
    }
    
    for seq in sequences:
        if seq == 'COMBINED_SEQ':
            continue
        else:
            for metric_group in x[0]['MotChallenge2DBox'][alg][seq]['pedestrian'].keys():
                for metric in x[0]['MotChallenge2DBox'][alg][seq]['pedestrian'][metric_group].keys():
                    if metric == 'HOTA(0)':
                        metrics['HOTA'] += x[0]['MotChallenge2DBox'][alg][seq]['pedestrian'][metric_group][metric]
                    if metric == 'MOTA':
                        metrics['MOTA'] += x[0]['MotChallenge2DBox'][alg][seq]['pedestrian'][metric_group][metric]
                    if metric == 'MOTP':
                        metrics['MOTP'] += x[0]['MotChallenge2DBox'][alg][seq]['pedestrian'][metric_group][metric]
                    if metric == 'IDSW':
                        metrics['IDSW'] += x[0]['MotChallenge2DBox'][alg][seq]['pedestrian'][metric_group][metric]
                    if metric == 'IDF1':
                        metrics['IDF1'] += x[0]['MotChallenge2DBox'][alg][seq]['pedestrian'][metric_group][metric]
            
    for m in metrics.keys():
        metrics[m] /= (len(sequences) - 1)
    
    print()
    
    print(f'Method: {alg} - COMBINED_SEQ')
    # Print table of results
    for m in ['HOTA', 'IDSW', 'MOTA', 'MOTP', 'IDF1']:
        print(f'{m}\t|', end=' ')
        
    print()
    for metric_group in x[0]['MotChallenge2DBox'][alg]['COMBINED_SEQ']['pedestrian'].keys():
        for metric in x[0]['MotChallenge2DBox'][alg]['COMBINED_SEQ']['pedestrian'][metric_group].keys():
            if metric == 'HOTA(0)':
                # print(f'{metric}\t|', end=' ')
                print(f'{x[0]["MotChallenge2DBox"][alg]["COMBINED_SEQ"]["pedestrian"][metric_group][metric]:.2f}\t|', end=' ')
            if metric == 'MOTA':
                # print(f'{metric}\t|', end=' ')
                print(f'{x[0]["MotChallenge2DBox"][alg]["COMBINED_SEQ"]["pedestrian"][metric_group][metric]:.2f}\t|', end=' ')
            if metric == 'MOTP':
                # print(f'{metric}\t|', end=' ')
                print(f'{x[0]["MotChallenge2DBox"][alg]["COMBINED_SEQ"]["pedestrian"][metric_group][metric]:.2f}\t|', end=' ')
            if metric == 'IDSW':
                # print(f'{metric}\t|', end=' ')
                print(f'{x[0]["MotChallenge2DBox"][alg]["COMBINED_SEQ"]["pedestrian"][metric_group][metric]:.2f}\t|', end=' ')
            if metric == 'IDF1':
                # print(f'{metric}\t|', end=' ')
                print(f'{x[0]["MotChallenge2DBox"][alg]["COMBINED_SEQ"]["pedestrian"][metric_group][metric]:.2f}\t|', end=' ')
                
        
    print()
    
    print()
    print(f'Method: {alg} - average over sequences')
    # Print table of results
    for m in ['HOTA', 'IDSW', 'MOTA', 'MOTP', 'IDF1']:
        print(f'{m}\t|', end=' ')
        
    print()
    for m in ['HOTA', 'IDSW', 'MOTA', 'MOTP', 'IDF1']:
        print(f'{metrics[m]:.2f}\t|', end=' ')
        
    print()
    
    print()
        

