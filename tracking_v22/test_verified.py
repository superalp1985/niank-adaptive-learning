import unittest

import numpy as np
import torch

from tracking_v22.verified import VerifiedLearner, PracticeDirectory


class VerifiedChecks(unittest.TestCase):
    def test_single_harmful_trial_is_retained_for_more_practice(self):
        learner = VerifiedLearner(2, device="cpu")
        batch = learner.synthetic_batch(16)
        learner.train_batch(batch, 1)
        initial = {k:v.clone() for k,v in learner.model.state_dict().items()}
        before_step = next(iter(learner.optimizer.state.values()))["step"].clone()
        learner.begin_trial(torch.zeros(1, 33), 1, decision_step=10)
        learner.train_batch(batch, 1)
        with torch.no_grad():
            for head in learner.model.heads:
                head.bias.add_(30.)
        credit = learner.settle_trial(learner.synthetic_batch(16))
        self.assertFalse(credit["reverted"])
        self.assertTrue(any(not torch.equal(value, learner.model.state_dict()[key])
                            for key, value in initial.items()))
        self.assertGreater(next(iter(learner.optimizer.state.values()))["step"], before_step)
        self.assertEqual(learner.optimizer.param_groups[0]["lr"], .0008)
        self.assertEqual(learner.negative_trials, 1)

    def test_active_eviction_and_cold_return_keep_catalogue(self):
        memory = PracticeDirectory(capacity=3)
        for i in range(8):
            memory.add(i, .05+i*.01)
        self.assertEqual(len(memory.active), 3)
        self.assertEqual(len(memory.entries), 8)
        cold = next(i for i in memory.entries if i not in memory.active)
        memory.activate(cold, "recall")
        self.assertIn(cold, memory.active)
        self.assertEqual(len(memory.active), 3)
        self.assertEqual(len(memory.entries), 8)
        memory.credit(cold, -.2)
        self.assertLess(memory.entries[cold]["benefit"], 0)

    def test_rate_changes_only_after_repeated_evidence(self):
        learner = VerifiedLearner(2, device="cpu")
        for index in range(12):
            batch = learner.synthetic_batch(8)
            learner.begin_trial(torch.zeros(1, 33), 1, decision_step=index*64)
            learner.train_batch(batch, 1)
            with torch.no_grad():
                for head in learner.model.heads:
                    head.bias.add_(30.)
            learner.settle_trial(learner.synthetic_batch(8))
            if index < 11:
                self.assertEqual(learner.optimizer.param_groups[0]["lr"], .0008)
        self.assertEqual(learner.learning_trials, 12)
        self.assertLess(learner.optimizer.param_groups[0]["lr"], .0008)
        self.assertEqual(learner.rollbacks, 0)


if __name__ == "__main__":
    unittest.main()
