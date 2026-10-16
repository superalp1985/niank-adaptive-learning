"""Delayed readouts share attention; the existing rule still chooses the body."""
import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tracking_v23.neural import ActiveLearner, AttentivePredictor


class DelayedPredictor(AttentivePredictor):
    @classmethod
    def upgrade(cls, model, horizons):
        model.__class__ = cls
        parameter = next(model.parameters())
        model.delayed_heads = nn.ModuleList([
            nn.Linear(96, 27*7, device=parameter.device, dtype=parameter.dtype)
            for _ in horizons])
        for head in model.delayed_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        return model

    def delayed(self, x):
        probability = self.attention(x)
        scale = self.peripheral_floor+(1-self.peripheral_floor)*64*probability
        visual = x[:, :512].reshape(-1, 64, 8)*scale[..., None]
        hidden = self.encoder(torch.cat((visual.flatten(1), x[:, 512:]), 1))
        return torch.stack([head(hidden).reshape(-1, 27, 7)
                            for head in self.delayed_heads], dim=1)


class DelayedLearner(ActiveLearner):
    @classmethod
    def upgrade(cls, agent, config):
        if isinstance(agent, cls):
            return agent
        if not isinstance(agent, ActiveLearner):
            raise ValueError("V24 requires a complete active V23 source")
        device = next(agent.model.parameters()).device
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config["migration_seed"])
            if devices:
                torch.cuda.manual_seed_all(config["migration_seed"])
            DelayedPredictor.upgrade(agent.model, config["delay_horizons"])
            if agent.trial is not None:
                DelayedPredictor.upgrade(agent.trial["before"], config["delay_horizons"])
        new_parameters = list(agent.model.delayed_heads.parameters())
        options = {key: copy.deepcopy(value)
                   for key, value in agent.optimizer.param_groups[0].items() if key != "params"}
        agent.optimizer.add_param_group(dict(options, params=new_parameters))
        for key, value in agent.model.state_dict().items():
            if key not in agent.initial_model:
                agent.initial_model[key] = value.clone()
        agent.delay_config = {key: config[key] for key in (
            "delay_horizons", "delay_weights", "delay_warmup", "delay_gain", "delay_epochs")}
        agent.delayed_updates = agent.delayed_samples = 0
        agent.delayed_horizon_samples = [0]*len(config["delay_horizons"])
        agent.force_no_delayed = False
        agent.__class__ = cls
        return agent

    def train_delayed(self, rows):
        if not rows:
            return {}
        x = torch.stack([row["x"] for row in rows])
        action = torch.tensor([row["action"] for row in rows], device=self.device)
        indices = {value: i for i, value in enumerate(self.delay_config["delay_horizons"])}
        horizon = torch.tensor([indices[row["horizon"]] for row in rows], device=self.device)
        target = torch.as_tensor(np.stack([row["target"] for row in rows]), device=self.device)
        sample = torch.arange(len(rows), device=self.device)
        for _ in range(self.delay_config["delay_epochs"]):
            selected = self.model.delayed(x)[sample, horizon, action]
            loss = F.smooth_l1_loss(selected, target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 2.)
            self.optimizer.step()
            self.delayed_updates += 1
        self.delayed_samples += len(rows)
        counts = [sum(row["horizon"] == value for row in rows)
                  for value in self.delay_config["delay_horizons"]]
        self.delayed_horizon_samples = [
            old+new for old, new in zip(self.delayed_horizon_samples, counts)]
        return dict(loss=float(loss.detach()), samples=len(rows), horizon_samples=counts,
                    updates=self.delayed_updates)

    @torch.no_grad()
    def delayed_values(self, x, proprio):
        consequence = self.model.delayed(x)
        energy = (1+2*(1-proprio[:, 5]))[:, None, None]
        integrity = (1+4*(1-proprio[:, 6]))[:, None, None]
        value = (3*energy*consequence[..., 0]+5*integrity*consequence[..., 1]
                 - 2*integrity*consequence[..., 2]-5*integrity*consequence[..., 3]
                 + 3*energy*consequence[..., 4]+energy*consequence[..., 5]
                 - 2*integrity*consequence[..., 6])
        weight = x.new_tensor(self.delay_config["delay_weights"])
        return (value.clamp(-2, 2)*weight[None, :, None]).sum(1)

    @torch.no_grad()
    def act(self, x, cues, proprio, step, training=True, prior_gain=1.):
        original, selected, uncertainty = super().act(
            x, cues, proprio, step, training, prior_gain)
        if self.force_no_delayed or not self.delayed_updates:
            return original, selected, uncertainty
        body = original % 9
        value = self.delayed_values(x, proprio).reshape(-1, 3, 9)
        index = torch.arange(len(x), device=x.device)
        # The experienced long return controls gaze, while current body choice
        # is retained. This avoids rewarding a head command for changing its
        # accompanying body command at the same decision.
        mean = self.model(x).mean(0)
        energy = 1+2*(1-proprio[:, 5])
        integrity = 1+4*(1-proprio[:, 6])
        short = (energy[:, None]*(mean[..., 2]+3*mean[..., 6])
                 - integrity[:, None]*(2*mean[..., 4]+5*mean[..., 5])
                 + 3*mean[..., 7]+5*mean[..., 8])
        gain = self.delay_config["delay_gain"]*min(
            self.delayed_updates/self.delay_config["delay_warmup"], 1.)
        logits = (min(self.effective_updates()/80, 1.)*.8*short.clamp(-2, 2)
                  ).reshape(-1, 3, 9)[index, :, body]+gain*value[index, :, body]
        if training:
            temperature = max(.12, .4*np.exp(-self.updates/300))
            gaze = torch.distributions.Categorical(logits=logits/temperature).sample()
        else:
            gaze = logits.argmax(1)
        actions = gaze*9+body
        return actions, mean[index, actions], uncertainty
