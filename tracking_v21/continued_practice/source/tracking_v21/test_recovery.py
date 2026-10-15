"""Committed runtime checkpoints must restore learning state and factual output."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import torch

from tracking_v21 import recovery
from tracking_v21.run import Session


class RecoveryContracts(unittest.TestCase):
    def test_transaction_checkpoint_and_continuation_match(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((recovery.ROOT / "config.json").read_text())
            config.update(worlds=4, replay_budget=96)
            torch.manual_seed(882)
            torch.cuda.manual_seed_all(882)
            live = Session(root / "live", "reversible", 882000, config)
            restored = None
            try:
                recovery.make_durable(live)
                recovery.advance(live, 16, 1)
                copy_path = root / "restored"
                copy_path.mkdir()
                target = sqlite3.connect(copy_path / "events.sqlite")
                live.db.connection.backup(target)
                target.close()
                recovery.advance(live, 16, 1)
                expected = live.agent.state()
                restored = recovery.restore(copy_path)
                self.assertEqual(restored.step, 16)
                self.assertGreater(restored.agent.optimizers[0].state.__len__(), 0)
                recovery.advance(restored, 16, 1)
                a = np.load(live.output / "transitions_00001.npz")
                b = np.load(restored.output / "transitions_00001.npz")
                try:
                    for key in a.files:
                        if a[key].dtype.kind == "f":
                            np.testing.assert_allclose(a[key], b[key], rtol=1e-5,
                                                       atol=1e-7, err_msg=key)
                        else:
                            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
                finally:
                    a.close()
                    b.close()
                for key, module in expected.items():
                    for name, value in module.items():
                        torch.testing.assert_close(
                            value, restored.agent.state()[key][name], rtol=1e-5, atol=1e-7)
            finally:
                live.close()
                if restored:
                    restored.close()

    def test_uncommitted_rows_roll_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.sqlite"
            raw = sqlite3.connect(path)
            raw.execute("CREATE TABLE facts(value)")
            raw.commit()
            wrapped = recovery.DeferredCommit(raw)
            wrapped.execute("INSERT INTO facts VALUES(1)")
            wrapped.commit()
            wrapped.close()
            check = sqlite3.connect(path)
            try:
                self.assertEqual(check.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 0)
            finally:
                check.close()


if __name__ == "__main__":
    unittest.main()
