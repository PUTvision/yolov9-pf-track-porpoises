from dataclasses import dataclass
from typing import Tuple
import random


@dataclass
class KeyPointsResult:
    """
    A class to represent the result of a keypoints.

    Attributes
    ----------
    tongue : Tuple[int, int]
        The coordinates of the tongue.
    tail : Tuple[int, int]
        The coordinates of the tail.
    """
    tongue: Tuple[int, int]
    tail: Tuple[int, int]

    def get_tongue_tail(self):
        return self.tongue, self.tail
