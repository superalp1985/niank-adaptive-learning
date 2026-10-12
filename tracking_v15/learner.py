"""Temporal observations and detached factual risk allocation."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from tracking_v14.learner import Agent as Previous, advantages, network


def temporal_features(now, before, evidence, reset):
    delta = now - before
    delta[reset] = 0
    return np.concatenate((now, delta, evidence), axis=1).astype(np.float32)


def factual_risk_loss(logits, actions, labels):
    selected = logits[torch.arange(len(actions), device=actions.device), actions]
    return nn.functional.binary_cross_entropy_with_logits(selected, labels.detach())


class Agent(Previous):
    def __init__(self, device="cuda", mode="temporal"):
        super().__init__(80, 4, device)
        self.mode = mode
        self.risk = network(80, 18).to(device)
        with torch.no_grad():
            self.risk[-1].weight.zero_()
            self.risk[-1].bias.fill_(-3)
        self.risk_optimizer = torch.optim.Adam(self.risk.parameters(), lr=0.0007)
        self.replay_x = torch.zeros((16384, 80), device=device)
        self.replay_a = torch.zeros(16384, dtype=torch.long, device=device)
        self.replay_y = torch.zeros((16384, 2), device=device)
        self.replay_size = self.replay_cursor = self.risk_presentations = 0
        self.logged_corrections, self.logged_risks = [], []

    @torch.no_grad()
    def correction(self, features):
        if self.mode == "temporal":
            return features.new_zeros((len(features), 9))
        risks = self.risk(features.detach()).reshape(-1, 9, 2).sigmoid()
        return -(2 * risks[:, :, 0] + 4 * risks[:, :, 1])

    def expert_logits(self, features, groups):
        return super().expert_logits(features, groups) + self.correction(features)

    @torch.no_grad()
    def act(self, features, greedy=False):
        x = torch.as_tensor(features, device=self.device)
        router = Categorical(logits=self.router(x))
        groups = router.logits.argmax(1) if greedy else router.sample()
        correction = self.correction(x)
        logits = Previous.expert_logits(self, x, groups) + correction
        policy = Categorical(logits=logits)
        actions = policy.logits.argmax(1) if greedy else policy.sample()
        self.logged_corrections.append(correction.detach().cpu().numpy())
        risk = self.risk(x).reshape(-1, 9, 2).sigmoid()
        self.logged_risks.append(risk[torch.arange(len(actions), device=self.device),
                                    actions].cpu().numpy())
        return dict(actions=actions, groups=groups, action_logp=policy.log_prob(actions),
                    route_logp=router.log_prob(groups), value=self.value(x).squeeze(1),
                    correction=correction)

    def update(self, rollout, last_features, epochs=4):
        convert = lambda key: torch.as_tensor(rollout[key], device=self.device)
        x = convert("features")
        with torch.no_grad():
            last = self.value(torch.as_tensor(last_features, device=self.device)).squeeze(1)
            adv, target = advantages(convert("reward"), convert("done"), convert("value"), last)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        arrays = dict(x=x.flatten(0, 1), advantage=adv.flatten(), target=target.flatten(),
                      groups=convert("groups").flatten(), actions=convert("actions").flatten(),
                      old_action=convert("action_logp").flatten(),
                      old_route=convert("route_logp").flatten(),
                      next=convert("next_obs").flatten(0, 1),
                      correction=torch.as_tensor(np.asarray(self.logged_corrections),
                                                 device=self.device).flatten(0, 1))
        count = len(arrays["x"])
        losses = []
        for _ in range(epochs):
            permutation = torch.randperm(count, device=self.device)
            for start in range(0, count, 768):
                ids = permutation[start:start + 768]
                b = {key: item[ids] for key, item in arrays.items()}
                for group in range(self.count):
                    rows = torch.nonzero(b["groups"] == group, as_tuple=True)[0]
                    if not len(rows):
                        continue
                    logits = (self.experts[group](b["x"][rows].detach()) +
                              b["correction"][rows].detach())
                    policy = Categorical(logits=logits)
                    ratio = (policy.log_prob(b["actions"][rows]) - b["old_action"][rows]).exp()
                    objective = torch.minimum(ratio * b["advantage"][rows],
                                              ratio.clamp(0.8, 1.2) * b["advantage"][rows])
                    self._step(self.experts[group], self.expert_optimizers[group],
                               -objective.mean() - 0.012 * policy.entropy().mean())
                router = Categorical(logits=self.router(b["x"].detach()))
                ratio = (router.log_prob(b["groups"]) - b["old_route"]).exp()
                objective = torch.minimum(ratio * b["advantage"],
                                          ratio.clamp(0.8, 1.2) * b["advantage"])
                self._step(self.router, self.router_optimizer,
                           -objective.mean() - 0.012 * router.entropy().mean())
                self._step(self.value, self.value_optimizer,
                           0.5 * (self.value(b["x"].detach()).squeeze(1) -
                                  b["target"]).square().mean())
                predictive_loss = (self.predict(b["x"], b["actions"]) -
                                   (b["next"] - b["x"][:, :32])).square().mean()
                self._step(self.predictor, self.predictor_optimizer, predictive_loss)
                losses.append(float(predictive_loss.detach()))
                self.updates += 1
                self.predictor_updates += 1
                self.presentations += len(ids)
        labels = torch.stack((convert("collision").flatten().float(),
                              convert("impact").flatten().float()), 1)
        indices = (torch.arange(count, device=self.device) + self.replay_cursor) % 16384
        self.replay_x[indices] = arrays["x"].detach()
        self.replay_a[indices] = arrays["actions"]
        self.replay_y[indices] = labels
        self.replay_cursor = (self.replay_cursor + count) % 16384
        self.replay_size = min(self.replay_size + count, 16384)
        for _ in range(4):
            indices = torch.randint(self.replay_size, (768,), device=self.device)
            logits = self.risk(self.replay_x[indices].detach()).reshape(-1, 9, 2)
            loss = factual_risk_loss(logits, self.replay_a[indices], self.replay_y[indices])
            self._step(self.risk, self.risk_optimizer, loss)
            self.risk_presentations += 768
        return float(np.mean(losses))

    def state(self):
        return dict(**super().state(), risk=self.risk.state_dict())

    def parameters(self):
        return super().parameters() + sum(p.numel() for p in self.risk.parameters())
