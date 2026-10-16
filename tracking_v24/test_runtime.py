from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from tracking_v22.evaluate_stages import parameter_hash
from tracking_v23.evaluate import isolated_rng
from tracking_v23.persistence import backup_database
from tracking_v23.session import Session as OriginalSession

try:
    from tracking_v24.neural import DelayedLearner
    from tracking_v24.session import Session
except ImportError:
    DelayedLearner = Session = None


def configuration(mechanism="delayed"):
    return dict(arm="active", mechanism=mechanism, variant="verified", seed=241,
                migration_seed=241, world_seed=113, bodies=2, block=4, horizon=2,
                action_every=2, recent_epochs=2, recall_epochs=2, active_capacity=3,
                delay_horizons=[4, 8, 16], delay_weights=[.2, .3, .5],
                delay_warmup=4, delay_gain=.8, delay_epochs=1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA runtime")
class RuntimeChecks(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(Session, "V24 runtime is not implemented")
        self.quiet = redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        if hasattr(self, "quiet"):
            self.quiet.__exit__(None, None, None)

    def source(self, root):
        torch.manual_seed(241)
        torch.cuda.manual_seed_all(241)
        np.random.seed(241)
        random.seed(241)
        original = OriginalSession(root/"source", configuration())
        original.advance(8)
        original.checkpoint()
        original.close()
        return root/"source"/"events.sqlite"

    def test_upgrade_preserves_policy_and_original_optimizer(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as temporary:
            root = Path(temporary)
            source = self.source(root)
            old = OriginalSession.restore(source.parent)
            old_prediction = old.agent.model(old.x).detach().clone()
            old_action = old.agent.act(old.x, old.cues, old.proprio, old.step, False)[0]
            old_states = copy.deepcopy(old.agent.optimizer.state_dict())
            old.close()
            new = Session.fork(source, root/"delayed", configuration())
            try:
                self.assertTrue(torch.equal(new.agent.model(new.x), old_prediction))
                self.assertTrue(torch.equal(
                    new.agent.act(new.x, new.cues, new.proprio, new.step, False)[0], old_action))
                new_states = new.agent.optimizer.state_dict()
                for key, values in old_states["state"].items():
                    for name, expected in values.items():
                        self.assertTrue(torch.equal(new_states["state"][key][name], expected))
                self.assertEqual(new.agent.delayed_updates, 0)
            finally:
                new.close()

    def test_delayed_learning_reaches_attention_without_changing_spike_width(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as temporary:
            root = Path(temporary)
            source = self.source(root)
            session = Session.fork(source, root/"delayed", configuration())
            try:
                before = session.agent.model.attention_score.weight.detach().clone()
                rows = [dict(x=session.x[i].clone(), action=i+9, horizon=4,
                             target=np.array([-.1, -.2, .5, 1, 0, 0, 0], np.float32))
                        for i in range(2)]
                session.agent.train_delayed(rows)
                session.agent.train_delayed(rows)
                self.assertFalse(torch.equal(before, session.agent.model.attention_score.weight))
                self.assertEqual(session.x.shape[1], 964)
                self.assertTrue(torch.equal(session.x[:, -32:], session.agent.spikes.trace/10))
            finally:
                session.close()

    def test_pending_delay_and_learning_resume_exactly(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as temporary:
            root = Path(temporary)
            source = self.source(root)
            live = Session.fork(source, root/"live", configuration())
            live.advance(8)
            self.assertGreater(live.delay.pending_count(), 0)
            live.checkpoint()
            backup_database(root/"live"/"events.sqlite", root/"resume"/"events.sqlite")
            live.advance(8)
            expected_hash = parameter_hash(live.agent)
            expected_pending = live.delay.pending_count()
            expected_targets = live.db.execute(
                "SELECT start,end,body,horizon,target FROM delayed_outcomes ORDER BY start,body,horizon"
            ).fetchall()
            live.close()
            resumed = Session.restore(root/"resume")
            try:
                resumed.advance(8)
                self.assertEqual(parameter_hash(resumed.agent), expected_hash)
                self.assertEqual(resumed.delay.pending_count(), expected_pending)
                self.assertEqual(resumed.db.execute(
                    "SELECT start,end,body,horizon,target FROM delayed_outcomes ORDER BY start,body,horizon"
                ).fetchall(), expected_targets)
            finally:
                resumed.close()

    def test_control_continuation_does_not_add_parameters_or_change_next_action(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as temporary:
            root = Path(temporary)
            source = self.source(root)
            backup_database(source, root/"reference"/"events.sqlite")
            reference = OriginalSession.restore(root/"reference")
            reference.learning_enabled = True
            with isolated_rng(29):
                reference.advance(4)
                expected = (reference.current_actions.copy(), parameter_hash(reference.agent))
            reference.close()
            control = Session.fork(source, root/"control", configuration("control"))
            try:
                with isolated_rng(29):
                    control.advance(4)
                np.testing.assert_array_equal(control.current_actions, expected[0])
                self.assertEqual(parameter_hash(control.agent), expected[1])
                self.assertFalse(hasattr(control.agent.model, "delayed_heads"))
            finally:
                control.close()

    def test_partial_next_frame_is_trimmed_before_resume(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as temporary:
            root = Path(temporary)
            source = self.source(root)
            live = Session.fork(source, root/"live", configuration())
            live.advance(8)
            live.checkpoint()
            committed = live.step
            live.advance(1)
            live.close()
            resumed = Session.restore(root/"live")
            try:
                self.assertEqual(resumed.step, committed)
                self.assertEqual(resumed.db.execute(
                    "SELECT count(*) FROM gaze_inputs WHERE step>=?", (committed,)
                ).fetchone()[0], 0)
                resumed.advance(2)
            finally:
                resumed.close()


if __name__ == "__main__":
    unittest.main()
