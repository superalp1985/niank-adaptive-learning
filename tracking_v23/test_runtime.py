import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import pickle
import random
import sqlite3
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from tracking_v22.evaluate_stages import parameter_hash
from tracking_v22.run import Session as LegacySession
from tracking_v23.evaluate import evaluate_checkpoint, isolated_rng
from tracking_v23.run import run, snapshot, source_identity, verify_manifest
from tracking_v23.session import Session, backup_database

torch.set_num_threads(4)


def configuration():
    return dict(variant="verified", seed=23, migration_seed=23, world_seed=113,
                bodies=2, additional_steps=8, block=4, horizon=2, action_every=2,
                checkpoint_every=4, milestone_every=4, stage_probe_steps=4,
                final_probe_steps=4, probe_checkpoint_every=2, stage_scene_seed=8113,
                final_scene_seed=98113, probe_seed=239, recent_epochs=2,
                recall_epochs=2, active_capacity=3, final_diagnostics=True)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA experiment runtime")
class RuntimeChecks(unittest.TestCase):
    def setUp(self):
        self.quiet = redirect_stdout(io.StringIO())
        self.quiet.__enter__()

    def tearDown(self):
        self.quiet.__exit__(None, None, None)

    def source(self, root, steps=8):
        torch.manual_seed(231)
        torch.cuda.manual_seed_all(231)
        np.random.seed(231)
        random.seed(231)
        path = root/"source"
        session = LegacySession(path, configuration())
        try:
            session.advance(steps)
            session.learning_enabled = False
            session.checkpoint()
        finally:
            session.close()
        return path/"events.sqlite"

    def test_baseline_fork_preserves_all_delayed_state_and_next_step(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as directory:
            root = Path(directory)
            source = self.source(root)
            backup_database(source, root/"legacy"/"events.sqlite")
            old = LegacySession.restore(root/"legacy")
            old.learning_enabled = True
            values = {key: getattr(old, key).clone() for key in
                      ("x", "features", "proprio", "sensory", "context")}
            sensory, drive = old.agent.last_sensory.clone(), old.agent.last_drive.clone()
            with isolated_rng(1):
                old.advance(1)
                expected_worlds = [(tuple(w.body.position), tuple(w.body.velocity))
                                   for w in old.worlds]
                expected_error = old.agent.error.clone()
            old.close()
            new = Session.fork(source, root/"baseline", dict(configuration(), arm="baseline"))
            try:
                self.assertTrue(new.learning_enabled)
                for key, expected in values.items():
                    self.assertTrue(torch.equal(getattr(new, key), expected), key)
                self.assertTrue(torch.equal(new.agent.last_sensory, sensory))
                self.assertTrue(torch.equal(new.agent.last_drive, drive))
                with isolated_rng(1):
                    new.advance(1)
                self.assertEqual([(tuple(w.body.position), tuple(w.body.velocity))
                                  for w in new.worlds], expected_worlds)
                torch.testing.assert_close(new.agent.error, expected_error, rtol=1e-5, atol=1e-6)
            finally:
                new.close()

    def test_active_fork_preserves_history_dynamic_state_and_source_bytes(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as directory:
            root = Path(directory)
            source = self.source(root, steps=7)
            before = source_identity(source)
            old = LegacySession.restore(source.parent)
            old_drive = old.agent.last_drive.clone()
            old_sensory = old.agent.last_sensory.clone()
            ticks, facts = old.agent.spikes.ticks, old.db.execute("SELECT count(*) FROM facts").fetchone()[0]
            old.close()
            # Source restore above is a test fixture only; establish identity now.
            before = source_identity(source)
            new = Session.fork(source, root/"active", dict(configuration(), arm="active"))
            try:
                self.assertEqual(new.origin_step, 7)
                self.assertEqual(new.x.shape[1], 964)
                self.assertEqual(new.agent.spikes.ticks, ticks)
                self.assertTrue(torch.equal(new.agent.last_drive, old_drive))
                self.assertTrue(torch.equal(new.agent.last_sensory, old_sensory))
                self.assertEqual(new.db.execute("SELECT count(*) FROM facts").fetchone()[0], facts)
                for queue in new.pending:
                    for row in queue:
                        self.assertEqual(row["x"].shape[0], 964)
                        self.assertEqual(int(torch.count_nonzero(row["x"][512:900])), 0)
                for record in new.records:
                    self.assertEqual(record[0].shape[0], 964)
                recalled = new.recall(new.context.mean(0, keepdim=True))
                self.assertEqual(recalled[0].shape[1], 964)
                self.assertEqual(int(torch.count_nonzero(recalled[0][:, 512:900])), 0)
                new.advance(1)
                self.assertEqual(new.agent.spikes.ticks, ticks+100)
            finally:
                new.close()
            self.assertEqual(source_identity(source), before)

    def test_both_arms_exact_checkpoint_resume_with_learning(self):
        for arm in ("baseline", "active"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory(dir="E:/niank-temp") as directory:
                root = Path(directory)
                source = self.source(root)
                path = root/arm
                original = Session.fork(source, path, dict(configuration(), arm=arm))
                original.advance(4)
                original.checkpoint()
                backup_database(path/"events.sqlite", root/"restored"/"events.sqlite")
                original.advance(4)
                expected = dict(weights=parameter_hash(original.agent),
                                worlds=[(tuple(w.body.position), tuple(w.body.velocity))
                                        for w in original.worlds],
                                spikes=original.agent.spikes.membrane.clone(),
                                actions=original.current_actions.copy(),
                                choices=original.choices.copy(),
                                vision=copy.deepcopy(original.visual_totals))
                original.close()
                resumed = Session.restore(root/"restored")
                try:
                    resumed.advance(4)
                    self.assertEqual(parameter_hash(resumed.agent), expected["weights"])
                    self.assertEqual([(tuple(w.body.position), tuple(w.body.velocity))
                                      for w in resumed.worlds], expected["worlds"])
                    self.assertTrue(torch.equal(resumed.agent.spikes.membrane, expected["spikes"]))
                    np.testing.assert_array_equal(resumed.current_actions, expected["actions"])
                    self.assertEqual(resumed.choices, expected["choices"])
                    self.assertEqual(resumed.visual_totals, expected["vision"])
                finally:
                    resumed.close()

    def test_active_death_resets_width_head_context_and_spike_clock(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as directory:
            session = Session(Path(directory)/"active", dict(configuration(), arm="active"))
            try:
                session.worlds[0].head_yaw = .5
                session.worlds[0].energy = 0
                ticks = session.agent.spikes.ticks
                session.advance(1)
                self.assertEqual(session.metrics["deaths"], 1)
                self.assertEqual(session.x.shape, (2, 964))
                self.assertEqual(session.agent.spikes.ticks, ticks+100)
                self.assertTrue(torch.equal(session.x[0, 512:516], torch.zeros(4, device="cuda")))
                self.assertEqual(session.worlds[0].head_yaw, 0.)
                self.assertEqual(float(session.context[0, 26]), 1.)
                self.assertEqual(float(session.context[0, 27]), 1.)
                self.assertTrue(torch.equal(session.x[:, -32:], session.agent.spikes.trace/10.))
                self.assertEqual(len(session.pending[0]), 0)
            finally:
                session.close()

    def test_probe_is_frozen_resumable_and_rng_isolated(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as directory:
            root = Path(directory)
            source = self.source(root)
            config = dict(configuration(), arm="active")
            training = Session.fork(source, root/"training", config)
            checkpoint = root/"agent.pt"
            torch.save(training.agent, checkpoint)
            training.close()
            rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone(),
                   pickle.dumps(np.random.get_state()), pickle.dumps(random.getstate()))
            original = Session.advance
            calls = []

            def interrupted(session, count):
                if calls:
                    raise RuntimeError("injected probe interruption")
                calls.append(1)
                return original(session, count)

            output = root/"probe"
            with mock.patch.object(Session, "advance", interrupted):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    evaluate_checkpoint(checkpoint, output, config, 4, 98113)
            self.assertTrue(torch.equal(rng[0], torch.get_rng_state()))
            self.assertTrue(torch.equal(rng[1], torch.cuda.get_rng_state()))
            self.assertEqual(rng[2], pickle.dumps(np.random.get_state()))
            self.assertEqual(rng[3], pickle.dumps(random.getstate()))
            result = evaluate_checkpoint(checkpoint, output, config, 4, 98113)
            control = evaluate_checkpoint(checkpoint, root/"control", config, 4, 98113)
            self.assertTrue(result["learned_parameters_unchanged"])
            self.assertEqual(result["parameter_sha256"], control["parameter_sha256"])
            self.assertEqual(result["scenes"], control["scenes"])
            self.assertEqual(result["frozen"]["updates"], training.agent.updates)
            self.assertEqual(result["frozen"]["meta_updates"], training.agent.meta_updates)
            self.assertEqual(result["frozen"]["unsettled_learning_choices"], 0)
            self.assertTrue((output/"actual_sensor.gif").exists())
            self.assertTrue((output/"actual_input_0.png").exists())
            again = evaluate_checkpoint(checkpoint, output, config, 4, 98113)
            self.assertEqual(again, result)

    def test_frozen_head_hold_and_uniform_interventions(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as directory:
            root = Path(directory)
            config = dict(configuration(), arm="active")
            session = Session(root/"training", config)
            with torch.no_grad():
                for head in session.agent.model.heads:
                    head.bias.zero_()
                    # Learned-value outcomes favor left gaze, without changing the reflex.
                    bias = head.bias.reshape(27, 9)
                    bias[9:18, 6] = 1.
            session.agent.updates = session.agent.accepted_updates = 100
            checkpoint = root/"agent.pt"
            torch.save(session.agent, checkpoint)
            session.close()
            normal = evaluate_checkpoint(checkpoint, root/"normal", config, 4, 98113)
            hold = evaluate_checkpoint(checkpoint, root/"hold", config, 4, 98113, "head_hold")
            uniform = evaluate_checkpoint(checkpoint, root/"uniform", config, 4, 98113, "uniform")
            self.assertGreater(normal["frozen"]["vision"]["head_travel"], 0.)
            self.assertEqual(hold["frozen"]["vision"]["head_travel"], 0.)
            self.assertEqual(uniform["parameter_sha256"], normal["parameter_sha256"])
            self.assertEqual(hold["parameter_sha256"], normal["parameter_sha256"])

    def test_probe_recovers_before_its_first_checkpoint(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as directory:
            root = Path(directory)
            config = dict(configuration(), arm="active")
            session = Session(root/"training", config)
            checkpoint = root/"agent.pt"
            torch.save(session.agent, checkpoint)
            session.close()
            output = root/"probe"
            with mock.patch.object(Session, "checkpoint", side_effect=RuntimeError("first save")):
                with self.assertRaisesRegex(RuntimeError, "first save"):
                    evaluate_checkpoint(checkpoint, output, config, 2, 98113)
            result = evaluate_checkpoint(checkpoint, output, config, 2, 98113)
            self.assertEqual(result["frozen"]["step"], 2)
            self.assertEqual(len(list(root.glob("probe.interrupted_*"))), 1)

    def test_finite_paired_runner_resume_and_completed_rejection(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as directory:
            root = Path(directory)
            source = self.source(root)
            output = root/"paired"
            results = run(configuration(), output, source)
            for arm in ("baseline", "active"):
                path = output/arm
                training = json.loads((path/"training_only_summary.json").read_text())
                self.assertEqual(training["new_steps"], 8)
                self.assertEqual(training["continuation"]["frames"], 16)
                self.assertTrue((path/"learner_step_00000.pt").exists())
                self.assertTrue((path/"learner_step_00004.pt").exists())
                self.assertTrue((path/"learner_step_00008.pt").exists())
                self.assertEqual(set(results[arm]["stages"]), {"0", "4", "8"})
                self.assertEqual(results[arm]["training"]["unsettled_learning_choices"], 0)
            with self.assertRaisesRegex(RuntimeError, "Completed"):
                run(configuration(), output, source)
            # Recovery after the final probe completed but before marker writing.
            (output/"COMPLETE.json").unlink()
            (output/"active"/"COMPLETE.json").unlink()
            original = results["active"]["final_probe"]["scenes"]
            resumed = run(configuration(), output, source)
            self.assertEqual(resumed["active"]["final_probe"]["scenes"], original)

    def test_runner_recovers_source_backup_before_initial_upgrade(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as directory:
            root = Path(directory)
            source = self.source(root)
            output = root/"paired"
            config = configuration()
            snapshot(output/"baseline", dict(config, arm="baseline"), source_identity(source))
            backup_database(source, output/"baseline"/"events.sqlite")
            result = run(config, output, source, arm="baseline")
            self.assertEqual(result["baseline"]["training"]["origin_step"], 8)
            self.assertEqual(result["baseline"]["training"]["new_steps"], 12)

    def test_manifest_rejects_code_and_configuration_changes(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as directory:
            root = Path(directory)
            config = dict(configuration(), arm="baseline")
            snapshot(root/"manifest", config, {})
            manifest_path = root/"manifest"/"manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["sources"]["tracking_v23/vision.py"] = "0"*64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "source version"):
                verify_manifest(root/"manifest")


if __name__ == "__main__":
    unittest.main()
