"""Local predictive learning, fine-clock LIF state and delayed meta-control.

The retina and motor prior are explicitly engineered starting competences.
There is no object-coordinate input and no backpropagation through physics.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def retina(images):
    """Native-pixel cross morphology before angular feature aggregation."""
    rgb = images.permute(0, 3, 1, 2).float() / 255.
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    green = ((g - r).clamp_min(0) * (g - b).clamp_min(0) * 8).clamp(0, 1)
    red = ((r - g).clamp_min(0) * (r - b).clamp_min(0) * 5).clamp(0, 1)
    obstacle = ((b - g).clamp_min(0) * (r - g).clamp_min(0) * 12).clamp(0, 1)
    white = ((torch.minimum(r, torch.minimum(g, b)) - .58) * 8).clamp(0, 1)
    cross = torch.zeros_like(white)
    # Horizontal circular padding preserves objects crossing forward at u=0.
    for k in (3, 9, 21, 45):
        horizontal = F.avg_pool2d(F.pad(white, (k//2, k//2, 0, 0),
                                       mode="circular"), (1, k), stride=1)
        vertical = F.avg_pool2d(white, (k, 1), stride=1, padding=(k//2, 0))
        box = F.avg_pool2d(F.pad(vertical, (k//2, k//2, 0, 0),
                                mode="circular"), (1, k), stride=1)
        cross = torch.maximum(cross, (torch.minimum(horizontal, vertical) - box).relu())
    green_near = F.max_pool2d(F.pad(green, (12, 12, 0, 0), mode="circular"),
                              (1, 25), stride=1)
    green_near = F.max_pool2d(green_near, (25, 1), stride=1, padding=(12, 0))
    cross *= green_near
    target_columns = cross.amax(dim=2).squeeze(1)
    target = F.max_pool1d(target_columns[:, None], 16, stride=16).squeeze(1)
    green_area = F.avg_pool1d(green.mean(2), 16, stride=16).squeeze(1).sqrt()
    red_area = F.avg_pool1d(red.mean(2), 16, stride=16).squeeze(1).sqrt()
    obstacle_area = F.avg_pool1d(obstacle.mean(2), 16, stride=16).squeeze(1).sqrt()
    white_area = F.avg_pool1d(white.mean(2), 16, stride=16).squeeze(1).sqrt()
    upper = F.avg_pool1d(green[:, :, :256].mean(2), 16, stride=16).squeeze(1).sqrt()
    lower = F.avg_pool1d(green[:, :, 256:].mean(2), 16, stride=16).squeeze(1).sqrt()
    luminance = F.avg_pool1d(rgb.mean((1, 2))[:, None], 16, stride=16).squeeze(1)
    features = torch.stack((target, green_area, red_area, obstacle_area,
                            white_area, upper, lower, luminance), -1)
    angle = (torch.arange(64, device=images.device) + .5) * (2 * math.pi / 64)
    weights = (target - .025).relu().pow(3)
    total = weights.sum(1).clamp_min(1e-8)
    direction = torch.stack(((weights * angle.cos()).sum(1) / total,
                             (weights * angle.sin()).sum(1) / total), 1)
    return features, dict(direction=direction, target_strength=target.amax(1),
                           target_columns=target, red=red_area, obstacle=obstacle_area,
                           target_size=(green_area*weights).sum(1)/total)


class SpikeState:
    """Event-solved LIF with threshold times rounded to a fine integration grid.

    Input is held between fresh sensor frames. Analytic event times replace
    100 tiny GPU launches; this does not invent intermediate observations.
    """
    def __init__(self, bodies, width, dt=.0001, device="cpu"):
        self.dt, self.interval, self.tau, self.trace_tau = dt, .01, .012, .05
        self.membrane = torch.zeros(bodies, width, device=device)
        self.trace = torch.zeros_like(self.membrane)
        self.total_spikes = torch.zeros_like(self.membrane)
        self.ticks = 0

    def advance(self, drive, duration=.01):
        drive = drive.clamp(0, 3)
        possible = drive > 1
        first = self.tau * torch.log(
            ((drive - self.membrane).clamp_min(1e-9) /
             (drive - 1).clamp_min(1e-9)).clamp_min(1.))
        first = (first / self.dt).ceil().clamp_min(1) * self.dt
        first = torch.where(possible, first, torch.full_like(first, 10.))
        period = self.tau * torch.log((drive / (drive - 1).clamp_min(1e-9)).clamp_min(1.))
        period = (period / self.dt).ceil().clamp_min(1) * self.dt
        events = first[..., None] + period[..., None] * torch.arange(
            8, device=drive.device)
        valid = (events <= duration + 1e-8) & possible[..., None]
        count = valid.sum(-1)
        last = torch.where(valid, events, torch.zeros_like(events)).amax(-1)
        after = drive * (1 - torch.exp(-(duration - last) / self.tau))
        no_spike = drive + (self.membrane - drive) * math.exp(-duration / self.tau)
        self.membrane = torch.where(count > 0, after, no_spike).clamp(0, .99999)
        impulse = torch.where(valid, torch.exp(-(duration - events) / self.trace_tau),
                              torch.zeros_like(events)).sum(-1)
        self.trace = self.trace * math.exp(-duration / self.trace_tau) + impulse
        self.total_spikes += count
        self.ticks += round(duration / self.dt)
        return self.trace / 10.

    def reset(self, indices):
        self.membrane[indices] = 0
        self.trace[indices] = 0


class LocalPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.SiLU(),
                                     nn.Linear(128, 96), nn.SiLU())
        self.heads = nn.ModuleList([nn.Linear(96, 9 * 9) for _ in range(2)])

    def forward(self, x):
        h = self.encoder(x)
        return torch.stack([head(h).reshape(-1, 9, 9) for head in self.heads])


def choose_learning(scores, explore=True):
    if explore:
        return torch.distributions.Categorical(logits=scores * 4).sample()
    return scores.argmax(-1)


class Learner:
    latent_dim, spike_dim, context_dim = 24, 32, 33
    input_dim = 512 + 8 + latent_dim + spike_dim

    def __init__(self, bodies, device="cuda", variant="autonomous", seed=22):
        self.device, self.variant = torch.device(device), variant
        self.model = LocalPredictor(self.input_dim).to(device)
        self.state_map = nn.Linear(25, self.latent_dim).to(device)
        self.state_tau = nn.Parameter(torch.linspace(-2., 1., self.latent_dim, device=device))
        self.state_readout = nn.Linear(self.latent_dim, 9).to(device)
        self.meta = nn.Sequential(nn.Linear(self.context_dim, 48), nn.Tanh(),
                                  nn.Linear(48, 3)).to(device)
        nn.init.zeros_(self.meta[-1].weight)
        nn.init.zeros_(self.meta[-1].bias)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.0008)
        self.state_optimizer = torch.optim.Adam(
            list(self.state_map.parameters()) + [self.state_tau] +
            list(self.state_readout.parameters()), lr=.0005)
        self.meta_optimizer = torch.optim.Adam(self.meta.parameters(), lr=.002)
        self.latent = torch.zeros(bodies, self.latent_dim, device=device)
        self.spikes = SpikeState(bodies, self.spike_dim, device=device)
        self.spike_weights = torch.randn(self.spike_dim, 25, device=device) * .12
        self.feedback = torch.randn(9, self.spike_dim, device=device) / 3
        self.error = torch.zeros(bodies, 1, device=device)
        self.pretrace = torch.zeros(bodies, 25, device=device)
        self.last_sensory = None
        self.last_drive = None
        self.updates = self.state_updates = self.meta_updates = self.local_updates = 0
        self.trial = None
        self.prediction_history = []
        self.initial_model = copy.deepcopy(self.model.state_dict())
        self.initial_state = torch.cat([p.detach().flatten().clone() for p in
                                       self.state_map.parameters()] +
                                      [self.state_tau.detach().clone()])
        self.initial_spike_weights = self.spike_weights.clone()

    def make_context(self, proprio):
        rates = self.spikes.trace / 10.
        return torch.cat((self.latent, proprio[:, 3:8], self.error,
                          rates.mean(1, keepdim=True), rates.std(1, keepdim=True),
                          torch.ones_like(self.error)*min(self.updates/100, 1)), 1)

    def effective_updates(self):
        return self.updates

    def learning_cost(self, attempts):
        return .004 * attempts

    @torch.no_grad()
    def perceive(self, features, proprio, event_bins=None):
        sensory = torch.cat((features.mean(1), features.amax(1), proprio, self.error), 1)
        tau = .02 + F.softplus(self.state_tau)
        if self.last_sensory is not None:
            target = torch.tanh(self.state_map(self.last_sensory))
            alpha = 1 - torch.exp(-.01 / tau)
            self.latent += alpha * (target - self.latent)
            self.pretrace = (math.exp(-.01/.05) * self.pretrace +
                             (1-math.exp(-.01/.05))*self.last_sensory)
            if event_bins is None:
                self.spikes.advance(self.last_drive)
            else:
                for tick in range(10):
                    self.spikes.advance(self.last_drive, duration=.001)
                    # Body-event impulses arrive at their measured substep end.
                    self.spikes.membrane += event_bins[:, tick:tick+1] * .12
                    crossed = self.spikes.membrane >= 1
                    self.spikes.total_spikes += crossed
                    self.spikes.trace += crossed
                    self.spikes.membrane = torch.where(
                        crossed, torch.zeros_like(self.spikes.membrane),
                        self.spikes.membrane)
        rates = self.spikes.trace / 10.
        self.last_sensory = sensory.clone()
        self.last_drive = 1.5 + .9 * torch.tanh(F.linear(sensory, self.spike_weights))
        x = torch.cat((features.flatten(1), proprio, self.latent, rates), 1)
        context = self.make_context(proprio)
        return x, sensory, context

    @torch.no_grad()
    def act(self, x, cues, proprio, step, training=True, prior_gain=1.):
        predictions = self.model(x)
        mean = predictions.mean(0)
        uncertainty = predictions.std(0).mean((1, 2))
        angles = torch.arange(8, device=x.device) * math.pi/4
        vectors = torch.cat((torch.zeros(1, 2, device=x.device),
                             torch.stack((angles.cos(), angles.sin()), 1)))
        strength = (cues["target_strength"] * 4).clamp(0, 1)
        attraction = cues["direction"] @ vectors.T * strength[:, None]
        sector_angles = (torch.arange(64, device=x.device)+.5) * 2*math.pi/64
        alignment = (torch.stack((sector_angles.cos(), sector_angles.sin()), 1) @ vectors.T).relu().pow(4)
        obstacles = cues["obstacle"] @ alignment
        threats = cues["red"] @ alignment
        # The disclosed initial reflex is shared by comparison arms.
        reflex = 2.0 * attraction - 2.8 * obstacles - 4. * threats
        reflex[:, 0] -= .15
        integrity_need = 1. + 4. * (1-proprio[:, 6])
        energy_need = 1. + 2. * (1-proprio[:, 5])
        # Predicted consequences: target xy, size, threat, touch, impact,
        # contact, energy delta, integrity delta.
        value = (energy_need[:, None] * (mean[:, :, 2] + 3*mean[:, :, 6])
                 - integrity_need[:, None] * (2*mean[:, :, 4] + 5*mean[:, :, 5])
                 + 3*mean[:, :, 7] + 5*mean[:, :, 8])
        learned_gain = min(self.effective_updates()/80, 1.) * .8
        logits = prior_gain * reflex + learned_gain * value.clamp(-2, 2)
        if training:
            # Exploration is explicit and falls with actual local update count.
            temperature = max(.12, .4 * math.exp(-self.updates/300))
            action = torch.distributions.Categorical(logits=logits / temperature).sample()
        else:
            action = logits.argmax(1)
        return action, mean[torch.arange(len(x), device=x.device), action], uncertainty

    @staticmethod
    def outcome(cues, before, after):
        return torch.cat((cues["direction"], cues["target_strength"][:, None],
                          cues["red"].amax(1, keepdim=True),
                          after[:, 3:5], after[:, 7:8],
                          after[:, 5:7] - before[:, 5:7]), 1)

    def train_batch(self, batch, mode, epochs=1):
        if mode == 0:
            return dict(loss=0., state_loss=0.)
        x, action, y, sensory, old_latent, pretrace = batch
        for _ in range(epochs):
            prediction = self.model(x)
            selected = prediction[:, torch.arange(len(x), device=self.device), action]
            loss = F.smooth_l1_loss(selected, y[None].expand_as(selected))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 2.)
            self.optimizer.step()
            # One-step local state prediction, detached at the module boundary.
            tau = .02 + F.softplus(self.state_tau)
            updated = old_latent + (1-torch.exp(-.01/tau)) * (
                torch.tanh(self.state_map(sensory)) - old_latent)
            state_loss = F.smooth_l1_loss(self.state_readout(updated), y)
            self.state_optimizer.zero_grad()
            state_loss.backward()
            self.state_optimizer.step()
            with torch.no_grad():
                residual = (y - selected.mean(0)).clamp(-1, 1)
                eligibility = x[:, -self.spike_dim:].clamp(0, 2)
                modulation = residual @ self.feedback
                delta = (modulation * eligibility).T @ pretrace / len(x)
                self.spike_weights += .002 * delta.clamp(-.2, .2)
                self.spike_weights.clamp_(-2, 2)
            self.updates += 1
            self.state_updates += 1
            self.local_updates += 1
        return dict(loss=float(loss.detach()), state_loss=float(state_loss.detach()))

    def synthetic_batch(self, count):
        return (torch.randn(count, self.input_dim, device=self.device),
                torch.randint(9, (count,), device=self.device),
                torch.randn(count, 9, device=self.device),
                torch.randn(count, 25, device=self.device),
                torch.randn(count, self.latent_dim, device=self.device),
                torch.randn(count, 25, device=self.device))

    def begin_trial(self, context, mode, decision_step=0):
        if self.trial is not None:
            raise RuntimeError("Unsettled learning choice")
        self.trial = dict(before=copy.deepcopy(self.model).eval(),
                          context=context.detach(), mode=int(mode), updates=self.updates,
                          decision_step=decision_step)

    def settle_trial(self, future_batch, same_data=False, windows=None):
        if same_data:
            raise ValueError("Learning credit requires later fresh observations")
        if self.trial is None:
            return {}
        if windows is not None:
            mask = windows[:, 0] >= self.trial["decision_step"]
            if not bool(mask.any()):
                return {}
            future_batch = tuple(t[mask] for t in future_batch)
        x, actions, y, *_ = future_batch
        with torch.no_grad():
            idx = torch.arange(len(x), device=self.device)
            old = self.trial["before"](x)[:, idx, actions]
            new = self.model(x)[:, idx, actions]
            old_loss = F.smooth_l1_loss(old, y[None].expand_as(old))
            new_loss = F.smooth_l1_loss(new, y[None].expand_as(new))
            gain = ((old_loss-new_loss) / old_loss.clamp_min(.0001)).clamp(-1, 1)
            cost = self.learning_cost(self.updates-self.trial["updates"])
            score = gain - cost
        estimate = self.meta(self.trial["context"])[:, self.trial["mode"]].mean()
        meta_loss = F.smooth_l1_loss(estimate, score)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        self.meta_updates += 1
        result = dict(mode=self.trial["mode"], gain=float(gain),
                      score=float(score), future_loss=float(new_loss),
                      old_future_loss=float(old_loss), cost=cost,
                      validation_samples=len(x),
                      decision_step=self.trial["decision_step"],
                      validation_min_start=int(windows[mask, 0].min())
                      if windows is not None else None)
        self.trial = None
        return result

    @torch.no_grad()
    def select_mode(self, context, explore=True):
        if self.variant == "fixed":
            return 1
        scores = self.meta(context).mean(0, keepdim=True)
        if explore:
            scores += torch.rand_like(scores) * .08
        return int(choose_learning(scores, explore)[0])

    def parameter_change(self):
        return sum(float((v-self.initial_model[k]).square().sum())
                   for k, v in self.model.state_dict().items()) ** .5

    def state_parameter_change(self):
        state = torch.cat([p.detach().flatten() for p in self.state_map.parameters()]
                          + [self.state_tau.detach()])
        return float((state-self.initial_state).norm())

    def spike_parameter_change(self):
        return float((self.spike_weights-self.initial_spike_weights).norm())
