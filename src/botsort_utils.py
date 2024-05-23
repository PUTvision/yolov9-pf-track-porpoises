import numpy as np


def preds_to_botsort(predictions):
    ocsort_predictions = []
    
    for prediction in predictions:
        x1, y1, x2, y2 = prediction.xyxy
        conf = prediction.confidence
        
        ocsort_predictions.append([x1, y1, x2, y2, conf])
    
    if len(ocsort_predictions) == 0:
        return np.array([]).reshape(0, 5)
    
    return np.array(ocsort_predictions)
