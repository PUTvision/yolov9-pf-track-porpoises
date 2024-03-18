from enum import Enum


class TrackState(Enum):
    NEW = 1
    CONFIRMED = 2
    MISSING = 3
    DEAD = 4
    
    
    def is_new(self):
        return self == TrackState.NEW
    
    def is_confirmed(self):
        return self == TrackState.CONFIRMED
    
    def is_missing(self):
        return self == TrackState.MISSING
    
    def is_dead(self):
        return self == TrackState.DEAD

