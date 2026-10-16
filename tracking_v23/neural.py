"""Trainable peripheral-preserving attention and explicit V22 state migration."""
import copy
import math

import torch
from torch import nn

from tracking_v22.verified import VerifiedLearner
from tracking_v23.vision import EXTRA_DIM, INPUT_DIM, migrate_batch


class AttentivePredictor(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.SiLU(),
                                     nn.Linear(128, 96), nn.SiLU())
        self.heads = nn.ModuleList([nn.Linear(96, 27*9) for _ in range(2)])
        self.sector_map = nn.Linear(8, 24)
        self.condition_map = nn.Linear(4+8+24+32, 24)
        self.attention_score = nn.Linear(24, 1, bias=False)
        nn.init.zeros_(self.attention_score.weight)
        self.peripheral_floor = .25
        self.force_uniform = False

    def attention(self, x):
        visual = x[:, :512].reshape(-1, 64, 8)
        condition = torch.cat((x[:, 512:516], x[:, 900:]), 1)
        score = self.attention_score(torch.tanh(
            self.sector_map(visual)+self.condition_map(condition)[:, None])).squeeze(-1)
        probability = torch.softmax(score, dim=1)
        if self.force_uniform:
            probability = torch.ones_like(probability)/64
        return probability

    def forward(self, x):
        probability = self.attention(x)
        scale = self.peripheral_floor+(1-self.peripheral_floor)*64*probability
        visual = x[:, :512].reshape(-1, 64, 8)*scale[..., None]
        h = self.encoder(torch.cat((visual.flatten(1), x[:, 512:]), 1))
        return torch.stack([head(h).reshape(-1, 27, 9) for head in self.heads])


def expand_tensor(name, value):
    if name == "encoder.0.weight":
        result = value.new_zeros(value.shape[0], INPUT_DIM)
        result[:, :512] = value[:, :512]
        result[:, 512+EXTRA_DIM:] = value[:, 512:]
        return result
    if name.startswith("heads."):
        return value.repeat((3,)+(1,)*(value.ndim-1))
    return value.clone()


def migrate_predictor(old):
    if isinstance(old, AttentivePredictor):
        return old
    parameter = next(old.parameters())
    model = AttentivePredictor().to(device=parameter.device, dtype=parameter.dtype)
    state = model.state_dict()
    for name, value in old.state_dict().items():
        state[name] = expand_tensor(name, value)
    model.load_state_dict(state)
    model.train(old.training)
    return model


def migrate_optimizer(old_optimizer, old_model, model):
    """Preserve all Adam group options, step counts and first/second moments."""
    old_names = {id(parameter): name for name, parameter in old_model.named_parameters()}
    new_parameters = dict(model.named_parameters())
    groups, assigned = [], set()
    for group in old_optimizer.param_groups:
        options = {key: copy.deepcopy(value) for key, value in group.items()
                   if key != "params"}
        names = [old_names[id(parameter)] for parameter in group["params"]]
        options["params"] = [new_parameters[name] for name in names]
        assigned.update(names)
        groups.append(options)
    groups[0]["params"] += [p for name, p in new_parameters.items() if name not in assigned]
    optimizer = type(old_optimizer)(groups)
    for old_parameter, state in old_optimizer.state.items():
        name = old_names[id(old_parameter)]
        optimizer.state[new_parameters[name]] = {
            key: (expand_tensor(name, value) if torch.is_tensor(value) and value.ndim
                  else value.clone() if torch.is_tensor(value) else copy.deepcopy(value))
            for key, value in state.items()}
    return optimizer


class ActiveLearner(VerifiedLearner):
    input_dim = INPUT_DIM

    def __init__(self, bodies, device="cuda", variant="verified", seed=23):
        super().__init__(bodies, device=device, variant=variant, seed=seed)
        self.model = AttentivePredictor().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.0008)
        self.initial_model = copy.deepcopy(self.model.state_dict())

    @classmethod
    def upgrade(cls, agent, seed=23):
        if isinstance(agent, cls):
            return agent
        device = next(agent.model.parameters()).device
        devices = [device.index or 0] if device.type == "cuda" else []
        # Additional initialization must not perturb the continued training RNG.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if devices:
                torch.cuda.manual_seed_all(seed)
            old_model = agent.model
            model = migrate_predictor(old_model)
            agent.optimizer = migrate_optimizer(agent.optimizer, old_model, model)
            initial = copy.deepcopy(model.state_dict())
            for name, value in agent.initial_model.items():
                initial[name] = expand_tensor(name, value)
            agent.initial_model = initial
            agent.model = model
            if agent.trial is not None:
                agent.trial["before"] = migrate_predictor(agent.trial["before"])
        agent.__class__ = cls
        return agent

    def train_batch(self, batch, mode, epochs=1):
        return super().train_batch(migrate_batch(batch), mode, epochs)

    @torch.no_grad()
    def act(self, x, cues, proprio, step, training=True, prior_gain=1.):
        predictions = self.model(x)
        mean = predictions.mean(0)
        uncertainty = predictions.std(0).mean((1, 2))
        angles = torch.arange(8, device=x.device)*math.pi/4
        vectors = torch.cat((torch.zeros(1, 2, device=x.device),
                             torch.stack((angles.cos(), angles.sin()), 1)))
        strength = (cues["target_strength"]*4).clamp(0, 1)
        attraction = cues["direction"] @ vectors.T*strength[:, None]
        sector_angles = (torch.arange(64, device=x.device)+.5)*2*math.pi/64
        alignment = (torch.stack((sector_angles.cos(), sector_angles.sin()), 1)
                     @ vectors.T).relu().pow(4)
        reflex = 2*attraction-2.8*(cues["obstacle"] @ alignment)-4*(cues["red"] @ alignment)
        reflex[:, 0] -= .15
        integrity_need = 1+4*(1-proprio[:, 6])
        energy_need = 1+2*(1-proprio[:, 5])
        value = (energy_need[:, None]*(mean[:, :, 2]+3*mean[:, :, 6])
                 - integrity_need[:, None]*(2*mean[:, :, 4]+5*mean[:, :, 5])
                 + 3*mean[:, :, 7]+5*mean[:, :, 8])
        self.prior_fraction = max(.45, 1.-max(0, self.accepted_updates-96)/512)
        # Each head choice receives exactly the same disclosed body reflex.
        logits = (prior_gain*self.prior_fraction*reflex.repeat(1, 3)
                  + min(self.effective_updates()/80, 1.)*.8*value.clamp(-2, 2))
        if training:
            temperature = max(.12, .4*math.exp(-self.updates/300))
            action = torch.distributions.Categorical(logits=logits/temperature).sample()
        else:
            action = logits.argmax(1)
        return action, mean[torch.arange(len(x), device=x.device), action], uncertainty
