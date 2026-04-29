import numpy as np
from rrt_base_3d import SimpleRRT


class RRTStar(SimpleRRT):
    def __init__(self, bounds_min, bounds_max, step_len=0.1, max_iters=1000, goal_tolerance=0.1, radius=0.2):
        super().__init__(bounds_min, bounds_max, step_len, max_iters, goal_tolerance)
        self.radius = float(radius)

    def plan(self, start, goal):
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)

        nodes = [start]
        parents = [-1]
        costs = [0.0]

        for _ in range(self.max_iters):
            rand = np.random.uniform(self.bounds_min, self.bounds_max)
            dists = [np.linalg.norm(n - rand) for n in nodes]
            nearest = int(np.argmin(dists))
            direction = rand - nodes[nearest]
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

            if np.linalg.norm(new_node - goal) < self.goal_tolerance:
                return self._extract(nodes, parents, new_idx)

        return None
