import numpy as np 
import math 
from rrt_star import RRTStar



class InformedRRTStar(RRTStar):
    def __init__(
        self,
        bounds_min,
        bounds_max,
        step_len=0.1,
        max_iters=1000,
        goal_tolerance=0.1,
        radius=0.2,
        sample_attempts=30,
    ):
        super().__init__(bounds_min, bounds_max, step_len, max_iters, goal_tolerance, radius=radius)
        self.sample_attempts = int(max(1, sample_attempts))

    def plan(self, start, goal):
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        nodes = [start]
        parents = [-1]
        costs = [0.0]

        c_min = float(np.linalg.norm(goal - start))
        c_best = float("inf")
        best_idx = None

        for _ in range(self.max_iters):
            if np.isfinite(c_best):
                sample = self._sample_informed(start, goal, c_min, c_best)
            else:
                sample = np.random.uniform(self.bounds_min, self.bounds_max)

            dists = [np.linalg.norm(n - sample) for n in nodes]
            nearest = int(np.argmin(dists))
            direction = sample - nodes[nearest]
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            new_node = nodes[nearest] + direction * self.step_len

            neighbor_idxs = [
                i for i, n in enumerate(nodes)
                if np.linalg.norm(n - new_node) < self.radius
            ]
            best_parent = nearest
            best_cost = costs[nearest] + np.linalg.norm(new_node - nodes[nearest])
            for i in neighbor_idxs:
                cand_cost = costs[i] + np.linalg.norm(new_node - nodes[i])
                if cand_cost < best_cost:
                    best_cost = cand_cost
                    best_parent = i

            nodes.append(new_node)
            parents.append(best_parent)
            costs.append(best_cost)

            new_idx = len(nodes) - 1
            for i in neighbor_idxs:
                cand_cost = best_cost + np.linalg.norm(nodes[i] - new_node)
                if cand_cost < costs[i]:
                    parents[i] = new_idx
                    costs[i] = cand_cost

            dist_goal = np.linalg.norm(new_node - goal)
            if dist_goal < self.goal_tolerance:
                cand_best = best_cost + dist_goal
                if cand_best < c_best:
                    c_best = cand_best
                    best_idx = new_idx

        if best_idx is not None:
            path = self._extract(nodes, parents, best_idx)
            if np.linalg.norm(path[-1] - goal) > 1e-6:
                path.append(goal.copy())
            return path

        return None

    def _sample_informed(self, start, goal, c_min, c_best):
        if not np.isfinite(c_best) or c_best <= c_min + 1e-9:
            return np.random.uniform(self.bounds_min, self.bounds_max)

        center = 0.5 * (start + goal)
        a1 = (goal - start) / max(c_min, 1e-9)
        C = self._build_rotation(a1)
        r1 = c_best / 2.0
        r2 = math.sqrt(max(c_best * c_best - c_min * c_min, 0.0)) / 2.0
        L = np.diag([r1, r2, r2])

        for _ in range(self.sample_attempts):
            x_ball = self._sample_unit_ball()
            sample = C @ (L @ x_ball) + center
            if np.all(sample >= self.bounds_min) and np.all(sample <= self.bounds_max):
                return sample

        return np.random.uniform(self.bounds_min, self.bounds_max)

    @staticmethod
    def _sample_unit_ball():
        vec = np.random.normal(size=3)
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            vec = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        vec = vec / norm
        radius = np.random.random() ** (1.0 / 3.0)
        return vec * radius

    @staticmethod
    def _build_rotation(a1):
        a1 = np.asarray(a1, dtype=np.float64)
        a1 = a1 / max(np.linalg.norm(a1), 1e-9)
        v = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(a1, v))) > 0.9:
            v = np.array([0.0, 1.0, 0.0])
        a2 = v - np.dot(v, a1) * a1
        a2 = a2 / max(np.linalg.norm(a2), 1e-9)
        a3 = np.cross(a1, a2)
        return np.column_stack([a1, a2, a3])
