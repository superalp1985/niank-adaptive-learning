"""Behavioral checks for temporal state and delayed learning decisions."""
import unittest

import numpy as np
import torch

from tracking_v22.neural import SpikeState, Learner, retina, choose_learning


class NeuralChecks(unittest.TestCase):
    def test_fine_clock_and_order(self):
        a = SpikeState(2, 16, dt=.0001)
        b = SpikeState(2, 16, dt=.00005)
        drive = torch.full((2, 16), 1.8)
        for _ in range(20):
            xa = a.advance(drive)
            xb = b.advance(drive)
        self.assertEqual(a.ticks, 2000)
        self.assertEqual(b.ticks, 4000)
        self.assertLess(float((xa - xb).abs().max()), .02)
        c = SpikeState(2, 16)
        c.advance(drive * 0)
        c.advance(drive * 2)
        d = SpikeState(2, 16)
        d.advance(drive * 2)
        d.advance(drive * 0)
        self.assertGreater(float((c.membrane - d.membrane).abs().max()), .001)

    def test_image_content_and_shape(self):
        image = torch.zeros(1, 512, 1024, 3, dtype=torch.uint8)
        image[:, 210:300, 250:340, 1] = 180
        image[:, 250:260, 270:320] = 240
        image[:, 230:280, 290:300] = 240
        features, cues = retina(image)
        self.assertEqual(tuple(features.shape), (1, 64, 8))
        self.assertGreater(float(cues["target_strength"][0]), 0)
        shifted, _ = retina(torch.roll(image, 128, dims=2))
        self.assertTrue(torch.allclose(shifted, torch.roll(features, 8, dims=1),
                                       atol=1e-5))

    def test_parameters_change_only_when_allowed(self):
        torch.manual_seed(10)
        learner = Learner(2, device="cpu")
        batch = learner.synthetic_batch(32)
        initial = {k: v.clone() for k, v in learner.model.state_dict().items()}
        learner.train_batch(batch, mode=0)
        self.assertTrue(all(torch.equal(v, learner.model.state_dict()[k])
                            for k, v in initial.items()))
        learner.train_batch(batch, mode=1)
        self.assertTrue(any(not torch.equal(v, learner.model.state_dict()[k])
                            for k, v in initial.items()))

    def test_controller_responds_to_measured_benefit(self):
        scores = torch.tensor([[0., -.2, .5], [0., .7, -.1]])
        self.assertEqual(choose_learning(scores, explore=False).tolist(), [2, 1])

    def test_later_observations_required_for_credit(self):
        learner = Learner(2, device="cpu")
        batch = learner.synthetic_batch(32)
        learner.begin_trial(torch.zeros(1, learner.context_dim), 1)
        learner.train_batch(batch, mode=1)
        with self.assertRaises(ValueError):
            learner.settle_trial(batch, same_data=True)
        credit = learner.settle_trial(learner.synthetic_batch(32), same_data=False)
        self.assertTrue(np.isfinite(credit["gain"]))


if __name__ == "__main__":
    unittest.main()
