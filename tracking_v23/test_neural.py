import copy
import unittest

import torch

from tracking_v22.verified import VerifiedLearner
from tracking_v23.neural import ActiveLearner, expand_tensor
from tracking_v23.vision import migrate_input

torch.set_num_threads(4)


class NeuralChecks(unittest.TestCase):
    def source(self):
        torch.manual_seed(71)
        agent = VerifiedLearner(2, device="cpu")
        agent.train_batch(agent.synthetic_batch(16), 1, epochs=2)
        return agent

    def test_model_all_head_choices_preserve_legacy_predictions(self):
        agent = self.source()
        x = torch.randn(12, 576)
        expected = agent.model(x).detach()
        ActiveLearner.upgrade(agent)
        actual = agent.model(migrate_input(x)).detach()
        for gaze in range(3):
            torch.testing.assert_close(actual[:, :, gaze*9:(gaze+1)*9], expected,
                                       rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(agent.model.attention(migrate_input(x)),
                                    torch.full((12, 64), 1/64)))

    def test_adam_moments_groups_and_steps_migrate_exactly(self):
        agent = self.source()
        agent.optimizer.param_groups[0]["lr"] = .000123
        old_parameters = dict(agent.model.named_parameters())
        old_states = {name: copy.deepcopy(agent.optimizer.state[p])
                      for name, p in old_parameters.items()}
        ActiveLearner.upgrade(agent)
        self.assertEqual(agent.optimizer.param_groups[0]["lr"], .000123)
        for name, parameter in agent.model.named_parameters():
            if name not in old_states:
                self.assertNotIn(parameter, agent.optimizer.state)
                continue
            for key, value in old_states[name].items():
                expected = expand_tensor(name, value) if value.ndim else value
                self.assertTrue(torch.equal(agent.optimizer.state[parameter][key], expected),
                                (name, key))
        self.assertEqual(len(agent.optimizer.param_groups[0]["params"]),
                         len(list(agent.model.parameters())))

    def test_migration_preserves_dynamic_state_and_rng(self):
        agent = self.source()
        before = copy.deepcopy(agent)
        rng = torch.get_rng_state().clone()
        ActiveLearner.upgrade(agent)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name in ("latent", "pretrace", "spike_weights", "error", "feedback", "state_tau"):
            self.assertTrue(torch.equal(getattr(agent, name), getattr(before, name)), name)
        self.assertEqual(agent.updates, before.updates)
        self.assertEqual(agent.accepted_updates, before.accepted_updates)

    def test_pending_before_trial_accepts_new_width_and_fresh_credit(self):
        agent = self.source()
        agent.begin_trial(torch.zeros(1, 33), 1, decision_step=10)
        x = torch.randn(8, 576)
        old = agent.trial["before"](x).detach()
        ActiveLearner.upgrade(agent)
        expanded = migrate_input(x)
        torch.testing.assert_close(agent.trial["before"](expanded)[:, :, :9], old)
        batch = list(agent.synthetic_batch(8))
        batch[0] = expanded
        agent.train_batch(tuple(batch), 1)
        windows = torch.tensor([[10, 20, 0, 0]]*8)
        result = agent.settle_trial(tuple(batch), windows=windows)
        self.assertEqual(result["validation_min_start"], 10)
        self.assertFalse(result["reverted"])
        self.assertIsNone(agent.trial)

    def test_attention_learns_from_consequences_and_depends_on_visual_state(self):
        agent = self.source()
        ActiveLearner.upgrade(agent)
        batch = list(agent.synthetic_batch(32))
        batch[1] = torch.arange(32) % 27
        x = batch[0]
        original = agent.model.attention(x).detach().clone()
        before = agent.model.attention_score.weight.detach().clone()
        agent.train_batch(tuple(batch), 1, epochs=8)
        attention = agent.model.attention(x).detach()
        self.assertFalse(torch.equal(agent.model.attention_score.weight, before))
        self.assertGreater(float((attention-original).abs().sum()), 1e-5)
        no_visual = x.clone()
        no_visual[:, :512] = 0
        no_state = x.clone()
        no_state[:, 512:516] = 0
        no_state[:, 900:] = 0
        self.assertGreater(float((attention-agent.model.attention(no_visual)).abs().sum()), 1e-5)
        self.assertGreater(float((attention-agent.model.attention(no_state)).abs().sum()), 1e-5)
        normal = agent.model(x).detach()
        agent.model.force_uniform = True
        self.assertGreater(float((normal-agent.model(x)).abs().sum()), 1e-5)
        agent.model.force_uniform = False
        no_detail = x.clone()
        no_detail[:, 516:900] = 0
        self.assertGreater(float((normal-agent.model(no_detail)).abs().sum()), 1e-5)

    def test_peripheral_weight_never_drops_below_floor(self):
        agent = ActiveLearner(2, device="cpu")
        with torch.no_grad():
            agent.model.attention_score.weight.fill_(100)
        attention = agent.model.attention(torch.randn(8, 964))
        scale = .25+.75*64*attention
        self.assertGreaterEqual(float(scale.min()), .25)
        torch.testing.assert_close(scale.mean(1), torch.ones(8))

    def test_body_reflex_identical_across_head_choices(self):
        agent = self.source()
        old = copy.deepcopy(agent)
        ActiveLearner.upgrade(agent)
        x = torch.randn(2, 576)
        cues = dict(target_strength=torch.ones(2)*.5, direction=torch.tensor([[1., 0.], [0., 1.]]),
                    obstacle=torch.zeros(2, 64), red=torch.zeros(2, 64))
        proprio = torch.zeros(2, 8)
        proprio[:, 5:7] = 1
        action, _, _ = agent.act(migrate_input(x), cues, proprio, 0, training=False)
        expected, _, _ = old.act(x, cues, proprio, 0, training=False)
        self.assertTrue(torch.equal(action % 9, expected))
        self.assertTrue(torch.equal(action // 9, torch.zeros(2, dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()
