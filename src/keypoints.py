from typing import List

import cv2
import onnxruntime as ort
import numpy as np

from src.keypoints_result import KeyPointsResult
from src.track import Track

class KeyPoints:
    def __init__(self, model_path: str, engine: str):
        self.model_path = model_path
        self.engine = engine
        
        providers = {
            'cuda': ['CUDAExecutionProvider'],
            'cpu': ['CPUExecutionProvider']
        }
        
        print(f'[LOG] Load {self.model_path} model with {self.engine} engine')
        
        self.session = ort.InferenceSession(self.model_path, providers=providers[self.engine])
        
        self.input = self.session.get_inputs()[0]
        self.input_name = self.input.name
        self.input_shape = self.input.shape
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]
        
        self.outputs = self.session.get_outputs()
        self.output_names = [output.name for output in self.outputs]
        
        self.max_offset = 10
        
        # model preheat
        _ = self.session.run(self.output_names, {self.input_name: np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)})
    
    @staticmethod
    def letterbox(im, new_shape=(128, 128), color=(0, 0, 0), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def _crop_roi(self, img: np.ndarray, track: Track) -> np.ndarray:
        x1, y1, x2, y2 = track.xyxy
        
        x1o = max(x1 - self.max_offset, 0)
        y1o = max(y1 - self.max_offset, 0)
        x2o = min(x2 + self.max_offset, img.shape[1])
        y2o = min(y2 + self.max_offset, img.shape[0])
        
        roi = img[y1o:y2o, x1o:x2o]
        
        return roi, (x1o - x1, y1o - y1)
    
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        img, ratio, dwdh = self.letterbox(img, auto=False, new_shape=(128, 128), color=(0,0,0))
        
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, ...]
        img = img.astype(np.float32)
        img /= 255.0
        
        return img, ratio, dwdh
    
    def _postprocess(self, batched_output: np.ndarray, rois, preprocess_rois) -> List[KeyPointsResult]:
        outputs = []
        
        for heatmap, (_, offsets), (_, ratio, dwdh) in zip(batched_output, rois, preprocess_rois):
            heatmap[heatmap < 0.1] = 0
            
            xo, yo = offsets
            
            tongue = np.unravel_index(np.argmax(heatmap[0]), heatmap[0].shape)
            tail = np.unravel_index(np.argmax(heatmap[1]), heatmap[1].shape)
            
            tongue = np.array([tongue[1], tongue[0]], dtype=np.float32)
            tail = np.array([tail[1], tail[0]], dtype=np.float32)
            
            if heatmap[0, int(tongue[1]), int(tongue[0])] < 0.1:
                tongue = None
            else:
                tongue -= np.array([dwdh[0], dwdh[1]], dtype=np.float32)
                tongue /= ratio
                tongue = (int(tongue[0] + xo), int(tongue[1] + yo))
                
            if heatmap[1, int(tail[1]), int(tail[0])] < 0.1:
                tail = None
            else:
                tail -= np.array([dwdh[0], dwdh[1]], dtype=np.float32)
                tail /= ratio
                tail = (int(tail[0] + xo), int(tail[1] + yo))
                
            outputs.append(KeyPointsResult(tongue, tail))
        
        return outputs
        
        
    def predict_and_update(self, img: np.ndarray, track_predictions: List[Track]):
        
        if len(track_predictions) == 0:
            return
        
        rois = [self._crop_roi(img, track) for track in track_predictions]
        preprocess_rois = [self._preprocess(roi[0]) for roi in rois]
        
        batched_input = np.concatenate([roi for roi, _, _ in preprocess_rois], axis=0)
        
        batched_output = self.session.run(self.output_names, {self.input_name: batched_input})[0]
        
        keypoints = self._postprocess(batched_output, rois, preprocess_rois)
        
        # for roi, key in zip(rois, keypoints):
        #     cv2.circle(roi, key.tongue, 4, (255, 0, 255), 2)
        #     cv2.circle(roi, key.tail, 4, (0, 165, 255), 2)
            
        #     cv2.imwrite('keypoints.jpg', roi)
        #     exit()
        
        for track, keypoints in zip(track_predictions, keypoints):
            track.keypoints = keypoints
