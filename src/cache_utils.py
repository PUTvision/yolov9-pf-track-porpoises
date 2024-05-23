from enum import Enum
from pathlib import Path

class CacheType(Enum):
    NO_USE = 0
    CREATE = 1
    LOAD = 2
    
class Cache:
    cache_base = './cache/'
    
    def __init__(self, use_cache: bool, name: str):
        if not use_cache:
            self.cache_type = CacheType.NO_USE
        else:
            self.cache_path = Path(self.cache_base + name)
        
            if self.cache_path.exists():
                self.cache_type = CacheType.LOAD
            else:
                self.cache_type = CacheType.CREATE
                self.cache_path.mkdir(parents=True, exist_ok=True)
                
    @property
    def use_cache(self):
        return self.cache_type == CacheType.CREATE or self.cache_type == CacheType.LOAD
    
    @property
    def load_model(self):
        return self.cache_type == CacheType.CREATE or self.cache_type == CacheType.NO_USE
        
    
