"""Local policy modules with factual consequence prediction and bounded relearning."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from tracking_v14.learner import advantages
from tracking_v17.learner import EyeNetwork
from tracking_v19.learner import Agent as PreviousAgent


HORIZON = 8


def outcome_utility(target):
    return (2 * target[..., 0] + 2 * target[..., 1] + 4 * target[..., 2] -
            2 * target[..., 3] - 5 * target[..., 4])


class Consequences(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = EyeNetwork(45)

    def forward(self, x, proprio):
        return self.network(x, proprio).reshape(-1, 9, 5)

    def estimates(self, x, proprio):
        raw = self(x, proprio)
        return torch.cat((raw[..., :1].tanh(), raw[..., 1:].sigmoid()), -1)


class Agent(PreviousAgent):
    def __init__(self, mode, device="cuda", guidance=.35):
        super().__init__(mode, device)
        self.guidance = float(guidance)
        self.consequences = Consequences().to(device)
        self.modules.append(self.consequences)
        self.optimizers.append(torch.optim.Adam(self.consequences.parameters(), lr=5e-4))
        self.consequence_presentations = self.replay_presentations = 0
        self.replay_policy_presentations = 0

    @torch.no_grad()
    def act(self, x, proprio, greedy=False):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        prop = torch.as_tensor(proprio, dtype=torch.float32, device=self.device)
        route = Categorical(logits=self.router(x, prop))
        groups = route.logits.argmax(1) if greedy else route.sample()
        logits = x.new_zeros((len(x), 9))
        attention = x.new_zeros((len(x), x.shape[-1]))
        for group, expert in enumerate(self.experts):
            rows = torch.nonzero(groups == group, as_tuple=True)[0]
            if len(rows):
                z, a = expert(x[rows], prop[rows], True)
                logits[rows], attention[rows] = z, a
        estimates = self.consequences.estimates(x, prop)
        guide = outcome_utility(estimates)
        guide = self.guidance * (guide - guide.mean(1, keepdim=True)).clamp(-.75, .75)
        policy = Categorical(logits=logits + guide)
        actions = policy.logits.argmax(1) if greedy else policy.sample()
        return dict(actions=actions, groups=groups, action_logp=policy.log_prob(actions),
                    route_logp=route.log_prob(groups), value=self.value(x, prop).squeeze(1),
                    attention=attention, guide_logits=guide,
                    outcome_predictions=estimates)

    def fit_consequences(self, batch, replay=False, batch_size=768):
        count = len(batch["actions"])
        if not count:
            return dict(consequence_loss=0., policy_relearn_loss=0.)
        arrays = {key: torch.as_tensor(value, device=self.device)
                  for key, value in batch.items()}
        order = torch.randperm(count, device=self.device)
        losses, policy_losses = [], []
        for start in range(0, count, batch_size):
            ids = order[start:start + batch_size]
            x, prop = arrays["features"][ids].float(), arrays["proprio"][ids].float()
            action, target = arrays["actions"][ids].long(), arrays["targets"][ids].float()
            raw = self.consequences(x, prop)[torch.arange(len(ids), device=self.device), action]
            loss = .5 * (raw[:, 0].tanh() - target[:, 0]).square().mean()
            loss += nn.functional.binary_cross_entropy_with_logits(raw[:, 1:], target[:, 1:])
            self.local_step(self.consequences, self.optimizers[-1], loss)
            losses.append(float(loss.detach()))
            self.consequence_presentations += len(ids)
            if replay:
                # Outcome-weighted imitation is bounded and never labels an unexecuted action.
                weight = (2 * target[:, 0] + 3 * target[:, 2] -
                          2 * target[:, 3] - 5 * target[:, 4]).clamp(0, 4)
                groups = arrays["groups"][ids].long()
                for group, expert in enumerate(self.experts):
                    rows = torch.nonzero(groups == group, as_tuple=True)[0]
                    rows = rows[weight[rows] > 0]
                    if not len(rows):
                        continue
                    policy = Categorical(logits=expert(x[rows], prop[rows]))
                    ploss = -.04 * (weight[rows] * policy.log_prob(action[rows])).mean()
                    optimizer = self.optimizers[group]
                    original_rates = [item["lr"] for item in optimizer.param_groups]
                    try:
                        for item, rate in zip(optimizer.param_groups, original_rates):
                            item["lr"] = .15 * rate
                        self.local_step(expert, optimizer, ploss)
                    finally:
                        for item, rate in zip(optimizer.param_groups, original_rates):
                            item["lr"] = rate
                    policy_losses.append(float(ploss.detach()))
                    self.replay_policy_presentations += len(rows)
                self.replay_presentations += len(ids)
        return dict(consequence_loss=float(np.mean(losses)),
                    policy_relearn_loss=float(np.mean(policy_losses)) if policy_losses else 0.)

    def update(self, data, last_features, last_proprio, epochs=2):
        t = lambda key: torch.as_tensor(data[key], device=self.device)
        with torch.no_grad():
            last = self.value(torch.as_tensor(last_features, device=self.device),
                              torch.as_tensor(last_proprio, device=self.device)).squeeze(1)
            adv, target = advantages(t("reward"), t("done"), t("value"), last,
                                     gamma=.995, lam=.97)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        arrays = dict(x=t("features").flatten(0, 1),
                      full=t("full_features").flatten(0, 1),
                      prop=t("proprio").flatten(0, 1),
                      actions=t("actions").flatten(), groups=t("groups").flatten(),
                      old_action=t("action_logp").flatten(), old_route=t("route_logp").flatten(),
                      advantage=adv.flatten(), target=target.flatten(),
                      guide=t("guide_logits").flatten(0, 1), starts=t("starts").flatten(),
                      bootstrap=t("bootstrap").flatten(0, 1).bool(),
                      next=t("next_rgb").flatten(0, 1).transpose(1, 2).float() / 255)
        count, losses = len(arrays["x"]), []
        for _ in range(epochs):
            order = torch.randperm(count, device=self.device)
            for start in range(0, count, 768):
                ids = order[start:start + 768]
                b = {key: value[ids] for key, value in arrays.items()}
                for group, expert in enumerate(self.experts):
                    rows = torch.nonzero(b["groups"] == group, as_tuple=True)[0]
                    if not len(rows):
                        continue
                    policy = Categorical(logits=expert(b["x"][rows], b["prop"][rows]) +
                                         b["guide"][rows])
                    ratio = (policy.log_prob(b["actions"][rows]) - b["old_action"][rows]).exp()
                    objective = torch.minimum(ratio * b["advantage"][rows],
                                              ratio.clamp(.8, 1.2) * b["advantage"][rows])
                    self.local_step(expert, self.optimizers[group],
                                    -objective.mean() - .012 * policy.entropy().mean())
                router = Categorical(logits=self.router(b["x"], b["prop"]))
                ratio = (router.log_prob(b["groups"]) - b["old_route"]).exp()
                objective = torch.minimum(ratio * b["advantage"],
                                          ratio.clamp(.8, 1.2) * b["advantage"])
                self.local_step(self.router, self.optimizers[4],
                                -objective.mean() - .012 * router.entropy().mean())
                self.local_step(self.value, self.optimizers[5],
                                .5 * (self.value(b["x"], b["prop"]).squeeze(1) -
                                      b["target"]).square().mean())
                for index, predictor in enumerate(self.predictors):
                    rows = torch.nonzero(b["bootstrap"][:, index], as_tuple=True)[0]
                    if not len(rows):
                        continue
                    full, prop, action = b["full"][rows], b["prop"][rows], b["actions"][rows]
                    normal = predictor(full, prop, action)
                    omitted = predictor(self.erased(full, b["starts"][rows]), prop, action)
                    weight = 1 + 3 * (b["next"][rows] - full[:, :3]).abs()
                    loss = (((normal - b["next"][rows]).square() +
                             (omitted - b["next"][rows]).square()) * weight).mean() / 2
                    self.local_step(predictor, self.optimizers[6 + index], loss)
                    self.predictor_presentations += 2 * len(rows)
                    losses.append(float(loss.detach()))
                self.presentations += len(ids)
                self.updates += 1
        return float(np.mean(losses)) if losses else 0.

    def state(self):
        return dict(**super().state(), consequences=self.consequences.state_dict())

    def load(self, state):
        expected = {"experts", "router", "value", "predictors", "consequences"}
        if set(state) not in (expected, expected - {"consequences"}):
            raise ValueError("Expected V19 or V20 state")
        for name, tensors in state.items():
            getattr(self, name).load_state_dict(tensors)
