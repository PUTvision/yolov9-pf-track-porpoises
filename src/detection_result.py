from dataclasses import dataclass
from typing import Tuple
import random

from src.keypoints_result import KeyPointsResult

@dataclass
class DetectionResult:
    """
    A class to represent the result of a detection.

    Attributes
    ----------
    label : str
        The label of the detected object.
    confidence : float
        The confidence of the detection.
    x : int
        The x coordinate of the center of the detected object.
    y : int
        The y coordinate of the center of the detected object.
    w : int
        The width of the detected object.
    h : int
        The height of the detected object.
    color : Tuple[int, int, int]
        The color of the bounding box.
    particle : bool
        The flag to indicate if the object is a particle.
    keypoints : KeyPointsResult
        The keypoints of the detected object.
    """

    label: str
    confidence: float
    x: int
    y: int
    w: int
    h: int
    color: Tuple[int, int, int] = (255, 0, 0)
    particle: bool = False
    keypoints: KeyPointsResult = None
    
    @property
    def xyxy(self):
        return (self.x - self.w//2, self.y - self.h//2, self.x + self.w//2, self.y + self.h//2)
    
    @property
    def xywh(self):
        return (self.x, self.y, self.w, self.h)
