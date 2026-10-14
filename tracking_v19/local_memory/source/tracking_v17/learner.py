"""Detached circular-eye modules and factual revisable sector retention."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from tracking_v14.learner import advantages
from tracking_v17.world import EYES


class EyeMemory:
    def __init__(self, worlds, mode):
        self.mode = mode
        self.previous = np.zeros((worlds, EYES, 3), np.float32)
        self.history = np.zeros_like(self.previous)
        self.age = np.zeros((worlds, EYES), np.float32)
        self.utility = np.zeros_like(self.age)
        self.updates = 0

    def features(self, rgb, proprio, reset):
        current = rgb.astype(np.float32) / 255
        shifts = np.rint(proprio[:, 2] * EYES / 2).astype(int)
        for i, shift in enumerate(shifts):
            if reset[i]:
                self.previous[i] = self.history[i] = 0
                self.age[i] = self.utility[i] = 0
            elif shift:
                for value in (self.previous, self.history, self.age, self.utility):
                    value[i] = np.roll(value[i], -int(shift), axis=0)
        fixed_gate = np.exp(-self.age / 32)
        gate = fixed_gate if self.mode == "fixed" else (
            fixed_gate * np.clip(1 + self.utility * 12, .10, 1.9))
        gate = np.clip(gate, 0, 1)
        x = np.concatenate((current, self.previous, self.history * gate[:, :, None],
                            gate[:, :, None], np.minimum(self.age / 128, 1)[:, :, None]), 2)
        # Retain old visual detail separately. Current RGB always remains untouched.
        change = np.abs(current - self.previous).mean(2)
        self.age = np.where(change > .07, 0, self.age + 1)
        keep = np.where(change > .07, .96 * gate, .65 * gate)
        self.history = keep[:, :, None] * self.history + (1 - keep[:, :, None]) * current
        self.previous = current.copy()
        return x.transpose(0, 2, 1).astype(np.float32)

    def feedback(self, local_gain):
        if self.mode == "revisable":
            self.utility = .9 * self.utility + .1 * np.clip(local_gain, -.08, .08)
            self.updates += 1


def circular_stack(inputs, channels=32):
    return nn.Sequential(nn.Conv1d(inputs, channels, 5, padding=2, padding_mode="circular"),
                         nn.ELU(), nn.Conv1d(channels, channels, 5, padding=2,
                                           padding_mode="circular"), nn.ELU())


class EyeNetwork(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.eye = circular_stack(11)
        self.attention = nn.Conv1d(32, 1, 1)
        angles = torch.arange(EYES) * (2 * torch.pi / EYES)
        self.register_buffer("phase", torch.stack((angles.cos(), angles.sin())))
        self.head = nn.Sequential(nn.Linear(134, 64), nn.Tanh(), nn.Linear(64, outputs))

    def forward(self, x, proprio, return_attention=False):
        sectors = self.eye(x.detach())
        attention = self.attention(sectors).squeeze(1).softmax(1)
        pooled = (sectors * attention[:, None]).sum(2)
        directional = torch.einsum("bcn,kn->bck", sectors * attention[:, None],
                                   self.phase).flatten(1)
        features = torch.cat((pooled, directional, sectors.mean(2), proprio.detach()), 1)
        result = self.head(features)
        return (result, attention) if return_attention else result


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(circular_stack(26), nn.Conv1d(32, 3, 1))

    def forward(self, x, proprio, action):
        additional = torch.cat((proprio.detach(), nn.functional.one_hot(action, 9).float()), 1)
        image = torch.cat((x.detach(), additional[:, :, None].expand(-1, -1, EYES)), 1)
        return self.layers(image).sigmoid()


class Agent:
    def __init__(self, mode, device="cuda"):
        self.mode, self.device = mode, device
        self.experts = nn.ModuleList([EyeNetwork(9) for _ in range(4)]).to(device)
        self.router, self.value = EyeNetwork(4).to(device), EyeNetwork(1).to(device)
        self.predictor = Predictor().to(device)
        self.modules = [*self.experts, self.router, self.value, self.predictor]
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=lr)
                           for m, lr in zip(self.modules, [3e-4] * 4 + [1.5e-4, 5e-4, 7e-4])]
        self.presentations = self.predictor_presentations = self.updates = 0

    @staticmethod
    def erased(x):
        result = x.detach().clone()
        result[:, 6:11] = 0
        return result

    def predict(self, x, proprio, actions):
        return (self.predictor(x, proprio, actions),
                self.predictor(self.erased(x), proprio, actions))

    @torch.no_grad()
    def act(self, x, proprio, greedy=False):
        x = torch.as_tensor(x, device=self.device)
        prop = torch.as_tensor(proprio, device=self.device)
        route = Categorical(logits=self.router(x, prop))
        groups = route.logits.argmax(1) if greedy else route.sample()
        logits = x.new_zeros((len(x), 9))
        attention = x.new_zeros((len(x), EYES))
        for group, expert in enumerate(self.experts):
            rows = torch.nonzero(groups == group, as_tuple=True)[0]
            if len(rows):
                z, a = expert(x[rows], prop[rows], True)
                logits[rows], attention[rows] = z, a
        policy = Categorical(logits=logits)
        actions = policy.logits.argmax(1) if greedy else policy.sample()
        return dict(actions=actions, groups=groups, action_logp=policy.log_prob(actions),
                    route_logp=route.log_prob(groups), value=self.value(x, prop).squeeze(1),
                    attention=attention)

    @staticmethod
    def local_step(module, optimizer, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite local objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(module.parameters(), .5)
        optimizer.step()

    def update(self, data, last_features, last_proprio, epochs=3):
        t = lambda key: torch.as_tensor(data[key], device=self.device)
        with torch.no_grad():
            last = self.value(torch.as_tensor(last_features, device=self.device),
                              torch.as_tensor(last_proprio, device=self.device)).squeeze(1)
            adv, target = advantages(t("reward"), t("done"), t("value"), last,
                                     gamma=.995, lam=.97)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        arrays = dict(x=t("features").flatten(0, 1), prop=t("proprio").flatten(0, 1),
                      actions=t("actions").flatten(), groups=t("groups").flatten(),
                      old_action=t("action_logp").flatten(), old_route=t("route_logp").flatten(),
                      advantage=adv.flatten(), target=target.flatten(),
                      next=t("next_rgb").flatten(0, 1).transpose(1, 2).float() / 255)
        count = len(arrays["x"])
        losses = []
        for _ in range(epochs):
            order = torch.randperm(count, device=self.device)
            for start in range(0, count, 768):
                ids = order[start:start + 768]
                b = {key: value[ids] for key, value in arrays.items()}
                for group, expert in enumerate(self.experts):
                    rows = torch.nonzero(b["groups"] == group, as_tuple=True)[0]
                    if not len(rows):
                        continue
                    policy = Categorical(logits=expert(b["x"][rows], b["prop"][rows]))
                    ratio = (policy.log_prob(b["actions"][rows]) - b["old_action"][rows]).exp()
                    obj = torch.minimum(ratio * b["advantage"][rows],
                                        ratio.clamp(.8, 1.2) * b["advantage"][rows])
                    self.local_step(expert, self.optimizers[group],
                                    -obj.mean() - .012 * policy.entropy().mean())
                router = Categorical(logits=self.router(b["x"], b["prop"]))
                ratio = (router.log_prob(b["groups"]) - b["old_route"]).exp()
                objective = torch.minimum(ratio * b["advantage"],
                                          ratio.clamp(.8, 1.2) * b["advantage"])
                self.local_step(self.router, self.optimizers[4],
                                -objective.mean() - .012 * router.entropy().mean())
                self.local_step(self.value, self.optimizers[5],
                                .5 * (self.value(b["x"], b["prop"]).squeeze(1) -
                                      b["target"]).square().mean())
                normal, omitted = self.predict(b["x"], b["prop"], b["actions"])
                weight = 1 + 3 * (b["next"] - b["x"][:, :3]).abs()
                loss = (((normal - b["next"]).square() +
                         (omitted - b["next"]).square()) * weight).mean() / 2
                self.local_step(self.predictor, self.optimizers[6], loss)
                losses.append(float(loss.detach()))
                self.presentations += len(ids)
                self.predictor_presentations += 2 * len(ids)
                self.updates += 1
        return float(np.mean(losses))

    def state(self):
        return dict(experts=self.experts.state_dict(), router=self.router.state_dict(),
                    value=self.value.state_dict(), predictor=self.predictor.state_dict())

    def load(self, state):
        for name, value in state.items():
            getattr(self, name).load_state_dict(value)

    def parameters(self):
        return sum(p.numel() for m in self.modules for p in m.parameters())
