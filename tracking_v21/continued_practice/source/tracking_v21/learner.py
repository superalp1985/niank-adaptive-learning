"""Signed relearning and a logged, reversible episodic policy correction."""

import numpy as np
import torch
from torch.distributions import Categorical

from tracking_v20.learner import Agent as PreviousAgent, outcome_utility


class Agent(PreviousAgent):
    def __init__(self, mode, device="cuda", guidance=.35):
        super().__init__(mode, device, guidance)
        self.negative_presentations = 0
        self.query = None

    @torch.no_grad()
    def act(self, x, proprio, greedy=False, external=None):
        if external is None and self.query is not None:
            external = self.query()
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        prop = torch.as_tensor(proprio, dtype=torch.float32, device=self.device)
        route = Categorical(logits=self.router(x, prop))
        groups = route.logits.argmax(1) if greedy else route.sample()
        logits = x.new_zeros((len(x), 9))
        attention = x.new_zeros((len(x), x.shape[-1]))
        for group, expert in enumerate(self.experts):
            rows = torch.nonzero(groups == group, as_tuple=True)[0]
            if len(rows):
                logits[rows], attention[rows] = expert(x[rows], prop[rows], True)
        estimates = self.consequences.estimates(x, prop)
        guide = outcome_utility(estimates)
        guide = self.guidance * (guide - guide.mean(1, keepdim=True)).clamp(-.75, .75)
        if external is not None:
            guide += torch.as_tensor(external, dtype=torch.float32, device=self.device)
        policy = Categorical(logits=logits + guide)
        actions = policy.logits.argmax(1) if greedy else policy.sample()
        return dict(actions=actions, groups=groups, action_logp=policy.log_prob(actions),
            route_logp=route.log_prob(groups), value=self.value(x, prop).squeeze(1),
            attention=attention, guide_logits=guide, outcome_predictions=estimates)

    def fit_consequences(self, batch, replay=False, batch_size=768):
        result = super().fit_consequences(batch, replay, batch_size)
        if not replay or self.mode != "reversible" or not len(batch["actions"]):
            return result
        targets = torch.as_tensor(batch["targets"], device=self.device)
        # Only actions actually executed in the past are suppressed.
        weights = (2 * targets[:, 3] + 5 * targets[:, 4] -
                   2 * targets[:, 0] - 3 * targets[:, 2]).clamp(0, 4)
        x = torch.as_tensor(batch["features"], device=self.device)
        prop = torch.as_tensor(batch["proprio"], device=self.device)
        action = torch.as_tensor(batch["actions"], device=self.device)
        groups = torch.as_tensor(batch["groups"], device=self.device)
        losses = []
        for group, expert in enumerate(self.experts):
            rows = torch.nonzero((groups == group) & (weights > 0), as_tuple=True)[0]
            for start in range(0, len(rows), batch_size):
                ids = rows[start:start + batch_size]
                policy = Categorical(logits=expert(x[ids], prop[ids]))
                # Unlikelihood is bounded below at zero, unlike signed log-probability.
                probability = policy.log_prob(action[ids]).exp().clamp(max=1 - 1e-5)
                loss = -.006 * (weights[ids] * torch.log1p(-probability)).mean()
                self.local_step(expert, self.optimizers[group], loss)
                self.negative_presentations += len(ids)
                losses.append(float(loss.detach()))
        result["negative_relearn_loss"] = float(np.mean(losses)) if losses else 0.
        return result
