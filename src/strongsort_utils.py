import numpy as np
import os 

from deep_sort.detection import Detection


def preds_to_strongsort(predictions):
    strongsort_predictions = []
    
    for pred in predictions:
        x1, y1, x2, y2 = pred.xyxy
        conf = pred.confidence
        
        w = x2 - x1
        h = y2 - y1
        
        strongsort_predictions.append(Detection(
            [x1, y1, w, h],
            conf,
            np.random.rand(128)
        ))
    
    return strongsort_predictions
