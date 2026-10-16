"""Episode-bounded physical consequences, without simulator reward or geometry."""
from collections import deque

import numpy as np


class ConsequenceWindows:
    def __init__(self, bodies, horizons):
        self.horizons = tuple(horizons)
        self.totals = np.zeros((bodies, 4), np.float64)
        self.queues = [deque() for _ in horizons]

    def start(self, step, x, actions, before, episodes):
        common = dict(start=int(step), x=x.detach().clone(),
                      actions=actions.detach().cpu().numpy().copy(),
                      before=np.array(before, np.float64, copy=True),
                      prefix=self.totals.copy(), episodes=list(episodes))
        for queue in self.queues:
            queue.append(dict(common, remaining=np.ones(len(episodes), bool)))

    def advance(self, step, after, events, terminals):
        self.totals += np.asarray(events, np.float64)
        terminals = np.asarray(terminals, bool)
        after = np.asarray(after, np.float64)
        rows = []
        for horizon, queue in zip(self.horizons, self.queues):
            # Usually only the oldest batch matures; scan younger batches only
            # when a terminal must flush that body's outstanding observations.
            candidates = list(queue) if terminals.any() else []
            if not candidates:
                candidates = []
                for item in queue:
                    if step-item["start"] < horizon:
                        break
                    candidates.append(item)
            for item in candidates:
                age = step-item["start"]
                selected = item["remaining"] & (terminals | (age >= horizon))
                for body in np.flatnonzero(selected):
                    delta = self.totals[body]-item["prefix"][body]
                    target = np.concatenate((
                        after[body]-item["before"][body],
                        [delta[0]/(age*.01)], delta[1:],
                        [float(terminals[body])])).astype(np.float32)
                    rows.append(dict(start=item["start"], end=int(step), body=int(body),
                                     episode=int(item["episodes"][body]), horizon=horizon,
                                     action=int(item["actions"][body]), x=item["x"][body],
                                     target=target))
                item["remaining"][selected] = False
            while queue and not queue[0]["remaining"].any():
                queue.popleft()
        return rows

    def pending_count(self):
        return sum(int(item["remaining"].sum()) for queue in self.queues for item in queue)
