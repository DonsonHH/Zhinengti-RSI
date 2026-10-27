"""Public map/commitment relations for the M8 actor-information experiment."""
from __future__ import annotations

import numpy as np

from .stress import StressDispatchEnv


class RelationDispatchEnv(StressDispatchEnv):
    def prepare(self):
        state = super().prepare()
        features = state['features']
        targets = np.asarray(self.targets)
        # [candidate target, neighboring target, edge features]
        point_cost = np.asarray([[self.costs[k][tuple(point)] for k in range(8)] for point in targets], dtype=np.float32) / 48
        edge = np.zeros((8, 8, 8), dtype=np.float32)
        edge[:, :, 0] = point_cost
        edge[:, :, 1:] = features[0][:, [0, 1, 2, 3, 6, 7, 8]][None, :, :]
        # [focal executor, candidate, team executor, team features]
        team = np.zeros((4, 8, 4, 8), dtype=np.float32)
        for focal in range(4):
            for actor, position in enumerate(self.env.positions):
                team[focal, :, actor, 0] = features[actor, :, 4] * 22 / 48
                goal = int(self.assigned[actor])
                if goal >= 0:
                    team[focal, :, actor, 1] = features[actor, goal, 4] * 22 / 48
                    team[focal, :, actor, 2] = point_cost[:, goal]
                    team[focal, :, actor, 3] = 1
                    team[focal, goal, actor, 4] = 1
                team[focal, :, actor, 5:7] = position / 23
                team[focal, :, actor, 7] = float(actor == focal)
        state['target_relations'] = edge
        state['team_relations'] = team
        return state
