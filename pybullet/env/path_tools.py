import numpy as np
import yaml



def generate_pos(max_radius: float) -> tuple[np.ndarray, np.ndarray] :
    start_pos = np.random.uniform(low=0.0, high=max_radius, size=3)
    goal_pos = np.random.uniform(low=0.0, high=max_radius, size=3)

    return start_pos, goal_pos



def in_bound(point , bounds_min , bounds_max) -> bool:
    return np.all(point >= bounds_min) and np.all(point <= bounds_max)

