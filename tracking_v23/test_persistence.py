import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

from tracking_v23.persistence import (
    atomic_json, backup_database, preserve_incomplete_bootstrap,
    read_complete, validate_database,
)


class PersistenceChecks(unittest.TestCase):
    def source(self, root):
        path = root/"source.sqlite"
        db = sqlite3.connect(path)
        try:
            db.execute("CREATE TABLE runtime(id INTEGER PRIMARY KEY,step INTEGER,sha256 TEXT,payload BLOB)")
            payload = b"complete runtime fixture"
            db.execute("INSERT INTO runtime VALUES(1,7,?,?)",
                       (hashlib.sha256(payload).hexdigest(), payload))
            db.commit()
        finally:
            db.close()
        return path

    def test_json_write_interruption_preserves_previous_complete_file(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as directory:
            path = Path(directory)/"manifest.json"
            atomic_json(path, dict(version=1))
            original = path.read_bytes()

            def interrupted(value, stream, **kwargs):
                stream.write('{"version":')
                raise OSError("simulated write interruption")

            with mock.patch("tracking_v23.persistence.json.dump", side_effect=interrupted):
                with self.assertRaisesRegex(OSError, "write interruption"):
                    atomic_json(path, dict(version=2))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(len(list(path.parent.glob("manifest.json.partial_*"))), 1)
            atomic_json(path, dict(version=3))
            self.assertEqual(json.loads(path.read_text())["version"], 3)

    def test_json_publish_interruption_never_creates_completion_marker(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as directory:
            path = Path(directory)/"COMPLETE.json"
            with mock.patch.object(Path, "replace", side_effect=OSError("simulated publish interruption")):
                with self.assertRaisesRegex(OSError, "publish interruption"):
                    atomic_json(path, dict(done=True))
            self.assertFalse(path.exists())
            self.assertEqual(len(list(path.parent.glob("COMPLETE.json.partial_*"))), 1)

    def test_invalid_complete_is_preserved_and_not_accepted(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as directory:
            path = Path(directory)/"COMPLETE.json"
            for raw in ('{"done":', '{}', '[]'):
                path.write_text(raw)
                self.assertIsNone(read_complete(path, lambda value: value["done"] is True))
                self.assertFalse(path.exists())
            self.assertEqual(len(list(path.parent.glob("COMPLETE.json.invalid_*"))), 3)
            atomic_json(path, dict(done=True))
            self.assertEqual(read_complete(path, lambda value: value["done"]), dict(done=True))

    def test_backup_interruption_never_publishes_partial_database(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as directory:
            root = Path(directory)
            source = self.source(root)
            destination = root/"arm"/"events.sqlite"
            before = source.read_bytes()

            def interrupted(status, remaining, total):
                raise RuntimeError("simulated backup interruption")

            with self.assertRaisesRegex(RuntimeError, "backup interruption"):
                backup_database(source, destination, progress=interrupted)
            self.assertFalse(destination.exists())
            self.assertEqual(len(list(destination.parent.glob("events.sqlite.partial_*"))), 1)
            backup_database(source, destination)
            self.assertEqual(validate_database(source), validate_database(destination))
            self.assertEqual(source.read_bytes(), before)
            with self.assertRaises(ValueError):
                backup_database(source, destination)
            with self.assertRaises(ValueError):
                backup_database(source, source)

    def test_legacy_incomplete_bootstrap_is_preserved_but_practiced_state_is_not_replaced(self):
        with tempfile.TemporaryDirectory(dir="E:/niank-temp") as directory:
            root = Path(directory)
            database = root/"events.sqlite"
            database.write_bytes(b"incomplete backup")
            preserve_incomplete_bootstrap(root)
            evidence = list(root.glob("interrupted_bootstrap_*"))
            self.assertEqual(len(evidence), 1)
            self.assertEqual((evidence[0]/"events.sqlite").read_bytes(), b"incomplete backup")
            self.assertFalse(database.exists())
            database.write_bytes(b"damaged later runtime")
            (root/"learner_step_00000.pt").write_bytes(b"milestone exists")
            with self.assertRaisesRegex(RuntimeError, "practiced database"):
                preserve_incomplete_bootstrap(root)
            self.assertEqual(database.read_bytes(), b"damaged later runtime")


if __name__ == "__main__":
    unittest.main()
