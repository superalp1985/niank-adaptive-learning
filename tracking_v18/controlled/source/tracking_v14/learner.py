"""Independent PPO expert/router updates plus factual physical prediction."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def network(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 96), nn.Tanh(), nn.Linear(96, 64),
                         nn.Tanh(), nn.Linear(64, outputs))


def advantages(rewards, dones, values, last_value, gamma=0.98, lam=0.95):
    result = torch.zeros_like(rewards)
    carry = torch.zeros_like(last_value)
    successor = last_value
    for step in reversed(range(len(rewards))):
        continuing = (~dones[step]).float()
        delta = rewards[step] + gamma * continuing * successor - values[step]
        carry = delta + gamma * lam * continuing * carry
        result[step] = carry
        successor = values[step]
    return result.detach(), (result + values).detach()


class Agent:
    def __init__(self, inputs=48, experts=4, device="cuda"):
        self.device, self.inputs, self.count = device, inputs, experts
        self.experts = nn.ModuleList([network(inputs, 9) for _ in range(experts)]).to(device)
        self.router, self.value = network(inputs, experts).to(device), network(inputs, 1).to(device)
        self.predictor = network(inputs + 9, 32).to(device)
        self.expert_optimizers = [torch.optim.Adam(module.parameters(), lr=0.0003)
                                  for module in self.experts]
        self.router_optimizer = torch.optim.Adam(self.router.parameters(), lr=0.00015)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=0.0005)
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=0.0007)
        self.updates, self.predictor_updates, self.presentations = 0, 0, 0

    def expert_logits(self, features, groups):
        output = features.new_zeros((len(features), 9))
        for index, expert in enumerate(self.experts):
            rows = torch.nonzero(groups == index, as_tuple=True)[0]
            if len(rows):
                output = output.index_copy(0, rows, expert(features[rows].detach()))
        return output

    @torch.no_grad()
    def act(self, features, greedy=False):
        x = torch.as_tensor(features, device=self.device)
        router = Categorical(logits=self.router(x))
        groups = router.logits.argmax(1) if greedy else router.sample()
        policy = Categorical(logits=self.expert_logits(x, groups))
        actions = policy.logits.argmax(1) if greedy else policy.sample()
        return dict(actions=actions, groups=groups, action_logp=policy.log_prob(actions),
                    route_logp=router.log_prob(groups), value=self.value(x).squeeze(1))

    def predict(self, features, actions):
        onehot = nn.functional.one_hot(actions, 9).float()
        return self.predictor(torch.cat((features.detach(), onehot), 1))

    def update(self, rollout, last_features, epochs=4):
        x = torch.as_tensor(rollout["features"], device=self.device)
        convert = lambda key: torch.as_tensor(rollout[key], device=self.device)
        reward, done, value = convert("reward"), convert("done"), convert("value")
        with torch.no_grad():
            last = self.value(torch.as_tensor(last_features, device=self.device)).squeeze(1)
            adv, target = advantages(reward, done, value, last)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        arrays = dict(x=x.flatten(0, 1), advantage=adv.flatten(), target=target.flatten(),
                      groups=convert("groups").flatten(), actions=convert("actions").flatten(),
                      old_action=convert("action_logp").flatten(),
                      old_route=convert("route_logp").flatten(),
                      next=convert("next_obs").flatten(0, 1))
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
                    policy = Categorical(logits=self.experts[group](b["x"][rows].detach()))
                    ratio = (policy.log_prob(b["actions"][rows]) -
                             b["old_action"][rows]).exp()
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
                prediction = self.value(b["x"].detach()).squeeze(1)
                self._step(self.value, self.value_optimizer,
                           0.5 * (prediction - b["target"]).square().mean())
                delta = b["next"] - b["x"][:, :32]
                predictive_loss = (self.predict(b["x"], b["actions"]) - delta).square().mean()
                self._step(self.predictor, self.predictor_optimizer, predictive_loss)
                losses.append(float(predictive_loss.detach()))
                self.updates += 1
                self.predictor_updates += 1
                self.presentations += len(ids)
        return float(np.mean(losses))

    @staticmethod
    def _step(model, optimizer, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite local loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    def state(self):
        return dict(experts=self.experts.state_dict(), router=self.router.state_dict(),
                    value=self.value.state_dict(), predictor=self.predictor.state_dict())

    def load(self, state):
        for name, weights in state.items():
            getattr(self, name).load_state_dict(weights)

    def parameters(self):
        return sum(p.numel() for model in (self.experts, self.router, self.value,
                                          self.predictor) for p in model.parameters())
