from typing import List

import cv2
import onnxruntime as ort
import numpy as np

from src.detection_result import DetectionResult


class YOLOv7:
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
        
        self.score_threshold = 0.1
        self.iou_threshold = 0.4
        self.conf_thresold = 0.6
        
        # model preheat
        _ = self.session.run(self.output_names, {self.input_name: np.zeros(self.input_shape, dtype=np.float32)})
    
    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
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

    
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        img, ratio, dwdh = self.letterbox(img, auto=False, new_shape=(self.input_height, self.input_width), color=(114, 114, 114))
        
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, ...]
        img = img.astype(np.float32)
        img /= 255.0
        
        return img, ratio, dwdh
    
    def _postprocess(self, outputs: np.ndarray, ratio: float, dwdh: float) -> List[DetectionResult]:
        
        detections = []
        
        for _,x0,y0,x1,y1,cls_id,score in outputs:
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round()
            cls_id = int(cls_id)

            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            
            bbox = self.xyxy_to_xywh(*box).astype(np.int32)
            
            if score < self.conf_thresold:
                continue
            
            if bbox[2] < 5 or bbox[3] < 5:
                continue
            
            detections.append(
                DetectionResult(
                    label=cls_id,
                    confidence=score,
                    x=bbox[0],
                    y=bbox[1],
                    w=bbox[2],
                    h=bbox[3]
                
                )
            )
            
        return detections
        
    def predict(self, img: np.ndarray) -> List[DetectionResult]:
        
        img, ratio, dwdh = self._preprocess(img)
        
        outputs = self.session.run(self.output_names, {self.input_name: img})[0]
        
        return self._postprocess(outputs, ratio, dwdh)

    def xywh_to_xyxy(self, x, y, w, h):
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2
        return np.array([x1, y1, x2, y2])
    
    def xyxy_to_xywh(self, x1, y1, x2, y2):
        x, y = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        return np.array([x, y, w, h])
