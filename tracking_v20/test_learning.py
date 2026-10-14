"""CPU contracts for delayed factual labels and separate local learning."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.distributions import Categorical

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tracking_v17.world import EYES
from tracking_v20.learner import Agent, HORIZON
from tracking_v20.run import Pending


def observation(seed=0, count=1):
    rng = np.random.default_rng(seed)
    return (rng.random((count, 11, EYES), dtype=np.float32),
            rng.normal(0, .1, (count, 6)).astype(np.float32))


def info(segment=0, progress=0., contact=0, safe_contact=0, collision=0, impact=0):
    return dict(segment=segment, progress=progress, contact=contact,
                safe_contact=safe_contact, collision=collision, impact=impact)


def push(pending, body, time, segment=0):
    x, prop = observation(body)
    pending.push(body, time, segment, x[0], x[0], prop[0], body % 9, body % 4)


def snapshots(modules):
    return [[value.detach().clone() for value in module.parameters()] for module in modules]


def unchanged(before, modules):
    return all(torch.equal(old, new) for values, module in zip(before, modules)
               for old, new in zip(values, module.parameters()))


class PendingContracts(unittest.TestCase):
    def test_horizon_uses_discounted_progress_and_factual_flag_maxima(self):
        pending = Pending(1)
        push(pending, 0, 0)
        for step in range(HORIZON):
            outcomes = pending.observe(0, info(progress=.12, collision=step == 2), False)
            if step < HORIZON - 1:
                self.assertEqual(outcomes, [])
        self.assertEqual(len(outcomes), 1)
        event = outcomes[0]
        self.assertEqual(event["horizon"], HORIZON)
        self.assertAlmostEqual(float(event["target"][0]),
                               .1 * sum(.95 ** step for step in range(HORIZON)), places=6)
        np.testing.assert_array_equal(event["target"][1:], [0, 0, 1, 0])
        self.assertEqual(len(pending.queues[0]), 0)

    def test_contact_closes_every_pending_window_before_next_goal(self):
        pending = Pending(1)
        push(pending, 0, 0)
        pending.observe(0, info(progress=.12), False)
        push(pending, 0, 1)
        events = pending.observe(0, info(progress=.24, contact=1, safe_contact=1), False)
        self.assertEqual([event["time"] for event in events], [0, 1])
        self.assertEqual([event["horizon"] for event in events], [2, 1])
        self.assertAlmostEqual(float(events[0]["target"][0]), .1 + .95 * .2, places=6)
        for event in events:
            np.testing.assert_array_equal(event["target"][1:], [1, 1, 0, 0])
        push(pending, 0, 2)
        events = pending.observe(0, info(progress=-.12), True)
        np.testing.assert_allclose(events[0]["target"], [-.1, 0, 0, 0, 0], atol=1e-7)

    def test_reset_closes_old_segment_and_new_segment_has_fresh_targets(self):
        pending = Pending(1)
        push(pending, 0, 0, segment=4)
        events = pending.observe(0, info(segment=4, impact=1), True)
        self.assertEqual(events[0]["segment"], 4)
        np.testing.assert_array_equal(events[0]["target"], [0, 0, 0, 0, 1])
        push(pending, 0, 1, segment=5)
        events = pending.observe(0, info(segment=5, contact=1, safe_contact=1), False)
        self.assertEqual(events[0]["segment"], 5)
        np.testing.assert_array_equal(events[0]["target"], [0, 1, 1, 0, 0])

    def test_observations_never_update_another_body(self):
        pending = Pending(2)
        push(pending, 0, 0, segment=2)
        push(pending, 1, 0, segment=8)
        first = pending.observe(0, info(segment=2, contact=1, safe_contact=1), False)
        self.assertEqual(first[0]["body"], 0)
        self.assertEqual(pending.queues[1][0]["horizon"], 0)
        np.testing.assert_array_equal(pending.queues[1][0]["target"], np.zeros(5))
        second = pending.observe(1, info(segment=8, collision=1, impact=1), True)
        self.assertEqual(second[0]["body"], 1)
        np.testing.assert_array_equal(second[0]["target"], [0, 0, 0, 1, 1])

    def test_unclosed_segment_change_is_rejected(self):
        pending = Pending(1)
        push(pending, 0, 0, segment=7)
        with self.assertRaisesRegex(RuntimeError, "crossed a segment"):
            pending.observe(0, info(segment=8), False)
        self.assertEqual(pending.queues[0][0]["horizon"], 0)

    def test_progress_target_is_clipped_at_window_close(self):
        pending = Pending(1)
        push(pending, 0, 0)
        events = pending.observe(0, info(progress=120), True)
        self.assertEqual(float(events[0]["target"][0]), 1.)


class LearningContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(20020)
        self.agent = Agent("random", device="cpu")

    def test_outcome_input_is_detached_and_other_modules_receive_no_gradient(self):
        x, prop = observation(3, 3)
        x = torch.tensor(x, requires_grad=True)
        prop = torch.tensor(prop, requires_grad=True)
        self.agent.consequences(x, prop).square().mean().backward()
        self.assertIsNone(x.grad)
        self.assertIsNone(prop.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in self.agent.consequences.parameters()))
        self.assertTrue(all(p.grad is None for module in self.agent.modules[:-1]
                            for p in module.parameters()))
        optimizer_parameters = [{id(parameter) for group in optimizer.param_groups
                                 for parameter in group["params"]}
                                for optimizer in self.agent.optimizers]
        self.assertEqual(optimizer_parameters[-1],
                         {id(p) for p in self.agent.consequences.parameters()})
        for index, parameters in enumerate(optimizer_parameters):
            for other in optimizer_parameters[index + 1:]:
                self.assertFalse(parameters & other)

    def test_ppo_reuses_logged_guide_after_outcome_head_changes(self):
        x, prop = observation(5, 4)
        with torch.no_grad():
            final = self.agent.router.head[-1]
            final.weight.zero_()
            final.bias.copy_(torch.tensor([1000., -1000., -1000., -1000.]))
        selected = self.agent.act(x, prop)
        with torch.no_grad():
            for parameter in self.agent.consequences.parameters():
                parameter.zero_()
            bias = self.agent.consequences.network.head[-1].bias.reshape(9, 5)
            bias[0, 0], bias[1, 0] = 10., -10.
        self.assertFalse(torch.allclose(selected["guide_logits"],
                                        self.agent.act(x, prop)["guide_logits"]))
        data = dict(features=x[None], full_features=x[None], proprio=prop[None],
                    reward=np.array([[0., .1, .2, .3]], np.float32),
                    done=np.zeros((1, 4), bool),
                    starts=np.zeros((1, 4), np.int64),
                    bootstrap=np.zeros((1, 4, 3), bool),
                    next_rgb=np.zeros((1, 4, EYES, 3), np.uint8))
        for key in ("actions", "groups", "action_logp", "route_logp", "value", "guide_logits"):
            data[key] = selected[key].numpy()[None]
        policies = []

        def record_policy(*args, **kwargs):
            policy = Categorical(*args, **kwargs)
            if kwargs["logits"].shape[-1] == 9:
                policies.append(policy)
            return policy

        with patch("tracking_v20.learner.Categorical", side_effect=record_policy), \
                patch.object(self.agent, "local_step"), \
                patch("torch.randperm", side_effect=lambda count, **kwargs: torch.arange(count)):
            self.agent.update(data, x, prop, epochs=1)
        self.assertEqual(len(policies), 1)
        recomputed = policies[0].log_prob(selected["actions"])
        torch.testing.assert_close(recomputed, selected["action_logp"], atol=1e-7, rtol=0)
        torch.testing.assert_close((recomputed - selected["action_logp"]).exp(),
                                   torch.ones(4), atol=1e-7, rtol=0)

    def test_nonpositive_replay_trains_outcomes_without_policy_imitation(self):
        targets = np.array([[0, 0, 0, 0, 0], [-.25, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0], [0, 1, 1, 0, 1],
                            [1, 0, 0, 1, 0]], np.float32)
        x, prop = observation(7, len(targets))
        batch = dict(features=x, proprio=prop, actions=np.arange(len(targets)) % 9,
                     groups=np.arange(len(targets)) % 4, targets=targets)
        before = snapshots(self.agent.modules)
        metrics = self.agent.fit_consequences(batch, replay=True, batch_size=2)
        self.assertTrue(unchanged(before[:-1], self.agent.modules[:-1]))
        self.assertFalse(unchanged(before[-1:], self.agent.modules[-1:]))
        self.assertTrue(all(p.grad is None for module in self.agent.modules[:-1]
                            for p in module.parameters()))
        self.assertEqual(self.agent.consequence_presentations, len(targets))
        self.assertEqual(self.agent.replay_presentations, len(targets))
        self.assertEqual(self.agent.replay_policy_presentations, 0)
        self.assertEqual(metrics["policy_relearn_loss"], 0.)
        self.assertTrue(np.isfinite(metrics["consequence_loss"]))

    def test_positive_replay_updates_only_executed_expert_and_outcome_head(self):
        x, prop = observation(9)
        batch = dict(features=x, proprio=prop, actions=np.array([2]),
                     groups=np.array([3]), targets=np.array([[.1, 1, 1, 0, 0]], np.float32))
        before = snapshots(self.agent.modules)
        self.agent.fit_consequences(batch, replay=True)
        for index, module in enumerate(self.agent.modules):
            with self.subTest(module=index):
                self.assertEqual(unchanged([before[index]], [module]),
                                 index not in (3, len(self.agent.modules) - 1))
        self.assertEqual(self.agent.replay_policy_presentations, 1)
        self.assertEqual(self.agent.replay_presentations, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
