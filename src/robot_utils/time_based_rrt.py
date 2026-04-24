import numpy as np 
import yaml
from dataclasses import dataclass 

import time 

from pybullet.env.path_tools import in_bound



class TimeRRTNode:
    def __init__(self , start , goal , timestamp , bounds_min , bounds_max):
        self.start = start 
        self.goal = goal 
        self.timestamp = timestamp 
        self.bounds_min = bounds_min 
        self.bounds_max = bounds_max 
        self.parent = None 
        self.children = []
    
    def distance(start , goal) -> float:
        return np.linalg.norm(start - goal) 
    
    
    

        