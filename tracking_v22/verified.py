"""V22-B: patient local trials, adaptive step size and revisable memory."""
from collections import deque
import math

import numpy as np
import torch

from tracking_v22.neural import Learner


class PracticeDirectory:
    def __init__(self, capacity=8):
        self.capacity = capacity
        self.entries = {}
        self.active = set()
        self.events = []
        self.clock = 0
        self.evictions = self.reactivations = 0

    def retention(self, key):
        entry = self.entries[key]
        age = max(self.clock-entry["last"], 0)
        return (entry["benefit"] + .15*math.sqrt(entry["error"]) +
                .02*math.log1p(entry["uses"]) - .003*age)

    def activate(self, key, reason):
        if key not in self.active:
            if reason == "recall":
                self.reactivations += 1
            if len(self.active) >= self.capacity:
                removed = min(self.active, key=self.retention)
                self.active.remove(removed)
                self.evictions += 1
                self.events.append(dict(kind="deactivate", fact=removed, clock=self.clock))
            self.active.add(key)
            self.events.append(dict(kind=reason, fact=key, clock=self.clock))
        self.entries[key]["last"] = self.clock

    def add(self, key, error):
        self.clock = key
        self.entries[key] = dict(error=error, benefit=0., uses=0, last=key)
        self.activate(key, "new")

    def select(self, rows, context):
        current = np.asarray(context).reshape(-1)
        cold = [row for row in rows if row[0] not in self.active]
        pool = cold if cold and np.random.random() < .25 else [
            row for row in rows if row[0] in self.active]
        if not pool:
            pool = rows
        scores = []
        for key, _, blob in pool:
            old = np.frombuffer(blob, np.float32)
            similarity = float(current@old / max(np.linalg.norm(current)*np.linalg.norm(old), 1e-6))
            scores.append(similarity + self.retention(key) + np.random.gumbel(scale=.10))
        selected = pool[int(np.argmax(scores))][0]
        self.activate(selected, "recall")
        self.entries[selected]["uses"] += 1
        return selected

    def credit(self, key, gain):
        entry = self.entries[key]
        entry["benefit"] = .7*entry["benefit"] + .3*gain
        self.events.append(dict(kind="revalue", fact=key, gain=gain, clock=self.clock))


class VerifiedLearner(Learner):
    def __init__(self, bodies, device="cuda", variant="verified", seed=22):
        super().__init__(bodies, device, variant, seed)
        self.accepted_updates = self.reverted_updates = self.rollbacks = 0
        self.lr_changes = []
        self.mode_counts = np.zeros(3, np.int64)
        self.mode_returns = np.zeros(3, np.float64)
        self.gain_window = deque(maxlen=12)
        self.negative_trials = 0
        self.learning_trials = 0

    def settle_trial(self, future_batch, same_data=False, windows=None):
        saved = self.trial
        result = super().settle_trial(future_batch, same_data, windows)
        if not result:
            return result
        attempts = self.updates-saved["updates"]
        old_lr = self.optimizer.param_groups[0]["lr"]
        self.accepted_updates += attempts
        new_lr = old_lr
        if attempts:
            self.learning_trials += 1
            self.negative_trials += int(result["gain"] < 0)
            self.gain_window.append(result["gain"])
            # User steering: an early failure can be adaptation. Accumulate
            # repeated fresh-window evidence; do not reject a single attempt.
            if len(self.gain_window) == 12 and self.learning_trials % 4 == 0:
                average = float(np.mean(self.gain_window))
                if average < -.03:
                    new_lr = max(.00005, old_lr*.85)
                elif average > .03:
                    new_lr = min(.003, old_lr*1.10)
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        mode = result["mode"]
        self.mode_counts[mode] += 1
        self.mode_returns[mode] += .2*(result["score"]-self.mode_returns[mode])
        if new_lr != old_lr:
            self.lr_changes.append(dict(step=result["decision_step"],
                                       before=old_lr, after=new_lr, gain=result["gain"]))
        result.update(reverted=False, learning_rate=new_lr,
                      attempted_updates=attempts, retained_updates=attempts,
                      completed_learning_trials=self.learning_trials,
                      recent_mean_gain=float(np.mean(self.gain_window)) if self.gain_window else 0.,
                      negative_trials=self.negative_trials,
                      interpretation="updates retained; not a claim of successful learning")
        return result

    @torch.no_grad()
    def select_mode(self, context, explore=True):
        neural = self.meta(context).mean(0).cpu().numpy()
        if explore and self.mode_counts.sum() < 12:
            return int(self.mode_counts.sum() % 3)
        bonus = .04*np.sqrt(np.log(self.mode_counts.sum()+2)/(self.mode_counts+1))
        score = neural + self.mode_returns + bonus
        if explore and np.random.random() < .30:
            return int(np.random.randint(3))
        return int(np.argmax(score))

    @torch.no_grad()
    def act(self, x, cues, proprio, step, training=True, prior_gain=1.):
        # Gradually reduce reliance on the fixed reflex only after substantial
        # local practice, retaining exploration during the adaptation interval.
        adaptation = max(0, self.accepted_updates-96)/512
        self.prior_fraction = max(.45, 1.-adaptation)
        return super().act(x, cues, proprio, step, training,
                           prior_gain=prior_gain*self.prior_fraction)

    @staticmethod
    def outcome(cues, before, after):
        target = Learner.outcome(cues, before, after)
        # Marker confidence is not object size. Use projected green area near
        # the marked region as the observable approach cue.
        target[:, 2] = cues["target_size"]
        return target

    def effective_updates(self):
        pending = self.updates-self.trial["updates"] if self.trial is not None else 0
        return self.accepted_updates+pending

    def learning_cost(self, attempts):
        # The learning budget comes before computational thrift in early practice.
        return 0. if self.learning_trials < 32 else .0005*attempts
