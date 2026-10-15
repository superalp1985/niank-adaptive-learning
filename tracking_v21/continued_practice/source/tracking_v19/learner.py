"""Budgeted reversible eye access with factual local ensemble evidence."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from tracking_v14.learner import advantages
from tracking_v17.learner import Agent as PreviousAgent, Predictor
from tracking_v17.world import EYES

GROUP = 16
MEMBERS = 3


def probe_starts(worlds, step):
    return ((int(step) + 3 * np.arange(worlds, dtype=np.int64)) % 8) * GROUP


def bootstrap_mask(body, time, segment):
    body, time, segment = np.broadcast_arrays(body, time, segment)
    with np.errstate(over="ignore"):
        value = ((body.astype(np.uint64)[..., None] + np.uint64(1)) *
                 np.uint64(0x9E3779B97F4A7C15))
        value = value ^ ((time.astype(np.uint64)[..., None] + np.uint64(1)) *
                         np.uint64(0xBF58476D1CE4E5B9))
        value = value ^ ((segment.astype(np.uint64)[..., None] + np.uint64(1)) *
                         np.uint64(0x94D049BB133111EB))
        value = value ^ (np.arange(1, MEMBERS + 1, dtype=np.uint64) *
                         np.uint64(0xD6E8FEB86659FD93))
        value = (value ^ (value >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> 27)) * np.uint64(0x94D049BB133111EB)
        value = value ^ (value >> 31)
    return (value >> 11) < np.uint64(int(.8 * (1 << 53)))


class EyeMemory:
    def __init__(self, worlds, mode, seed):
        if mode not in ("utility", "protected", "calibration"):
            raise ValueError("Unknown V19 memory mode")
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.previous = np.zeros((worlds, EYES, 3), np.float32)
        self.history = np.zeros_like(self.previous)
        self.age = np.zeros((worlds, EYES), np.float32)
        self.utility = np.zeros_like(self.age)
        self.deviation = np.zeros_like(self.age)
        self.selected = np.zeros((worlds, EYES), bool)
        self.explored = np.zeros_like(self.selected)
        self.promoted = np.zeros_like(self.selected)
        self.full_features = np.zeros((worlds, 11, EYES), np.float32)
        self.updates = 0

    def features(self, rgb, proprio, reset):
        reset = np.asarray(reset, bool)
        current = np.asarray(rgb, np.float32) / 255
        shifts = np.rint(np.asarray(proprio)[:, 2] * EYES / 2).astype(int)
        for body, shift in enumerate(shifts):
            if reset[body]:
                for value in (self.previous, self.history, self.age, self.utility,
                              self.deviation, self.selected):
                    value[body] = 0
            elif shift:
                for value in (self.previous, self.history, self.age, self.utility,
                              self.deviation, self.selected):
                    value[body] = np.roll(value[body], -int(shift), axis=0)
        old_selection = self.selected.copy()
        base_gate = np.exp(-self.age / 32)
        complete = np.concatenate(
            (current, self.previous, self.history * base_gate[:, :, None],
             base_gate[:, :, None], np.minimum(self.age / 128, 1)[:, :, None]), 2)
        self.full_features = complete.transpose(0, 2, 1).astype(np.float32, copy=True)
        score = self.utility.copy()
        if self.mode == "protected":
            score += self.deviation
        elif self.mode == "calibration":
            score.fill(0)
        # Random tie-breaking cannot displace an eye with a strictly larger score.
        rank = np.lexsort((self.rng.random(score.shape), -score), axis=1)
        self.selected.fill(False)
        self.explored.fill(False)
        for body, order in enumerate(rank):
            self.selected[body, order[:56]] = True
            exploration = self.rng.choice(order[56:], 8, replace=False)
            self.explored[body, exploration] = True
            self.selected[body, exploration] = True
        self.promoted = self.selected & ~old_selection
        self.promoted[reset] = False
        result = self.full_features.copy()
        result[:, 6:10] *= self.selected[:, None, :]
        # Access is reversible; unselected history still receives factual updates.
        change = np.abs(current - self.previous).mean(2)
        self.age = np.where(change > .07, 0, self.age + 1).astype(np.float32)
        keep = np.where(change > .07, .96 * base_gate, .65 * base_gate)
        self.history = (keep[:, :, None] * self.history +
                        (1 - keep[:, :, None]) * current).astype(np.float32)
        self.previous = current.copy()
        return result

    def feedback(self, scores, starts):
        if self.mode == "calibration":
            return
        scores = np.asarray(scores, np.float32)
        starts = np.asarray(starts, np.int64)
        if scores.shape != (len(self.utility), MEMBERS) or starts.shape != (len(scores),):
            raise ValueError("Expected one local score per body and ensemble member")
        if not np.isfinite(scores).all() or ((starts < 0) | (starts >= EYES)).any():
            raise ValueError("Invalid local feedback")
        mean, deviation = scores.mean(1), scores.std(1, ddof=0)
        # The intervened input group already owns the current-frame coordinates.
        for body, start in enumerate(starts):
            eyes = (int(start) + np.arange(GROUP)) % EYES
            self.utility[body, eyes] = .9 * self.utility[body, eyes] + .1 * mean[body]
            self.deviation[body, eyes] = (.9 * self.deviation[body, eyes] +
                                          .1 * deviation[body])
        self.updates += 1


class Agent(PreviousAgent):
    def __init__(self, mode, device="cuda"):
        super().__init__(mode, device)
        self.predictors = nn.ModuleList([Predictor() for _ in range(MEMBERS)]).to(device)
        del self.predictor
        self.modules = [*self.experts, self.router, self.value, *self.predictors]
        self.optimizers = self.optimizers[:6] + [
            torch.optim.Adam(member.parameters(), lr=7e-4) for member in self.predictors]

    @staticmethod
    def erased(x, starts):
        starts = torch.as_tensor(starts, dtype=torch.long, device=x.device)
        mask = (torch.arange(EYES, device=x.device)[None] - starts[:, None]) % EYES < GROUP
        result = x.detach().clone()
        result[:, 6:11] = result[:, 6:11].masked_fill(mask[:, None], 0)
        return result

    def predict(self, x, proprio, actions, starts):
        erased = self.erased(x, starts)
        return (torch.stack([p(x, proprio, actions) for p in self.predictors], 1),
                torch.stack([p(erased, proprio, actions) for p in self.predictors], 1))

    @torch.no_grad()
    def evidence(self, full_features, proprio, actions, starts, after, floor):
        if not np.isfinite(floor) or floor <= 0:
            raise ValueError("Evidence floor must be positive and finite")
        x = torch.as_tensor(full_features, dtype=torch.float32, device=self.device)
        prop = torch.as_tensor(proprio, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        starts = torch.as_tensor(starts, dtype=torch.long, device=self.device)
        label = torch.as_tensor(after, device=self.device).transpose(1, 2).float() / 255
        normal, omitted = self.predict(x, prop, actions, starts)
        local = ((torch.arange(EYES, device=self.device)[None] -
                  starts[:, None] + 4) % EYES < GROUP + 8).float()
        normal_errors = (normal - label[:, None]).square().mean(2)
        omitted_errors = (omitted - label[:, None]).square().mean(2)
        normal_local = (normal_errors * local[:, None]).sum(2) / (GROUP + 8)
        omitted_local = (omitted_errors * local[:, None]).sum(2) / (GROUP + 8)
        variance = normal.var(1, unbiased=False).mean(1)
        result = dict(
            full_mse=(normal.mean(1) - label).square().mean((1, 2)),
            omitted_mse=(omitted.mean(1) - label).square().mean((1, 2)),
            normal_local=normal_local, omitted_local=omitted_local,
            scores=(omitted_local - normal_local) / (omitted_local + normal_local + floor),
            prediction_disagreement=(variance * local).sum(1) / (GROUP + 8))
        return {key: value.cpu().numpy() for key, value in result.items()}

    def update(self, data, last_features, last_proprio, epochs=3):
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
                      starts=t("starts").flatten(),
                      bootstrap=t("bootstrap").flatten(0, 1).bool(),
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
        return dict(experts=self.experts.state_dict(), router=self.router.state_dict(),
                    value=self.value.state_dict(), predictors=self.predictors.state_dict())

    def load(self, state):
        if set(state) != {"experts", "router", "value", "predictors"}:
            raise ValueError("Use load_ancestor for a V17/V18 single-predictor state")
        for name, tensors in state.items():
            getattr(self, name).load_state_dict(tensors)

    def load_ancestor(self, state):
        for name in ("experts", "router", "value"):
            getattr(self, name).load_state_dict(state[name])
        for index, predictor in enumerate(self.predictors):
            predictor.load_state_dict(state["predictor"])
            if index:
                generator = torch.Generator(device="cpu").manual_seed(19000 + index)
                with torch.no_grad():
                    for param in predictor.parameters():
                        scale = .01 * max(float(param.std(unbiased=False)), .001)
                        noise = torch.randn(param.shape, generator=generator, dtype=param.dtype)
                        param.add_(noise.to(param.device) * scale)
