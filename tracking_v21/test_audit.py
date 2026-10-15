"""Audit temporal eligibility and retention accounting without new training."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from tracking_v21.analyze import memory_audit


class AuditContracts(unittest.TestCase):
    def fixture(self, path, past=6, benefit=.1):
        db = sqlite3.connect(path / "events.sqlite")
        try:
            db.executescript("""
                CREATE TABLE directory_events(body,time);
                CREATE TABLE memory_catalog(body,time,bucket,benefit,tests);
                CREATE TABLE memory_trials(body,start,end,enabled,baseline,actual,
                                           estimate,records);
                CREATE TABLE outcomes(body,time,horizon,targets);
                CREATE TABLE cold_recalls(body,time,recalled_at);
            """)
            db.execute("INSERT INTO directory_events VALUES(0,?)", (past,))
            db.execute("INSERT INTO memory_catalog VALUES(0,?,0,?,1)", (past, benefit))
            db.execute("INSERT INTO memory_trials VALUES(0,10,17,1,-.25,0,.5,?)",
                       (json.dumps([[past, 1., benefit]]),))
            for time in (past, 10):
                db.execute("INSERT INTO outcomes VALUES(0,?,8,?)",
                           (time, np.zeros(5, np.float32).tobytes()))
            db.commit()
        finally:
            db.close()

    def test_memory_can_mature_during_trial_before_later_query(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.fixture(path)
            result = memory_audit(path)
            self.assertEqual(result["trials_checked"], 1)

    def test_memory_maturing_after_trial_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.fixture(path, past=11)
            with self.assertRaises(AssertionError):
                memory_audit(path)

    def test_incorrect_retention_recurrence_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.fixture(path, past=0, benefit=.2)
            with self.assertRaises(AssertionError):
                memory_audit(path)


if __name__ == "__main__":
    unittest.main()
