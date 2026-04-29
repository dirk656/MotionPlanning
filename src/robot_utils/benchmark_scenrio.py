import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

@dataclass
class Scenario:
    start: np.ndarray
    goal: np.ndarray
    static_obs: Optional[np.ndarray] = None
    dynamic_fn: Optional[Callable[[float], Sequence]] = None
    point_cloud: Optional[np.ndarray] = None
    name: Optional[str] = None
    speed: float = 1.0
    static_obs_format: str = "min_size"
    goal_tolerance: float = 0.05
    sim_dt: float = 0.1
    max_sim_time: float = 12.0

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=np.float64)
        self.goal = np.asarray(self.goal, dtype=np.float64)
        if self.static_obs is not None:
            self.static_obs = np.asarray(self.static_obs, dtype=np.float64)
        if self.point_cloud is not None:
            self.point_cloud = np.asarray(self.point_cloud, dtype=np.float64)


@dataclass
class PlannerOutput:
    path: list
    success: bool
    planning_time: float
    extra: Dict = field(default_factory=dict)


class BasePlanner:
    def plan(self, scenario, current_t=0.0):
        raise NotImplementedError

    def replan(self, scenario, current_t=0.0):
        return self.plan(scenario, current_t=current_t)

    def is_path_valid(self, path, scenario, current_t=0.0):
        return True

    def reset(self, seed=None):
        _ = seed
