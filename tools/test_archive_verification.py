"""Parallel readback must never authorize cleanup after a missing/bad part."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import archive_snapshot as archive
from tools import verify_archive_parallel as verifier


class VerificationContracts(unittest.TestCase):
    def fixture(self, root):
        data = {f"test.zip.part{i:03d}": bytes([i]) * 256 for i in range(1, 4)}
        parts = [dict(name=n, bytes=len(v), sha256=hashlib.sha256(v).hexdigest())
                 for n, v in data.items()]
        manifest = dict(repository="superalp1985/niank-adaptive-learning", tag="test", parts=parts)
        archive.dump(root / "manifest.json", manifest)
        return manifest, data

    def test_failed_part_is_resumed_without_repeating_verified_downloads(self):
        with tempfile.TemporaryDirectory(dir=archive.ROOT / "cache") as tmp:
            root = Path(tmp)
            manifest, data = self.fixture(root)
            calls = []
            failing = {"test.zip.part002"}

            def remote(tag, name):
                calls.append(name)
                if name == "manifest.json":
                    return io.BytesIO(json.dumps(manifest).encode())
                if name in failing:
                    raise ConnectionError("simulated release disconnect")
                return io.BytesIO(data[name])

            with patch.object(verifier, "location", return_value=root), \
                 patch.object(verifier, "read_manifest", return_value=manifest), \
                 patch.object(verifier, "remote_part", side_effect=remote):
                with self.assertRaisesRegex(RuntimeError, "not verified"):
                    verifier.verify("test", 2)
                self.assertFalse((root / "verified.json").exists())
                failing.clear()
                verifier.verify("test", 2)
            self.assertEqual(calls.count("test.zip.part001"), 1)
            self.assertEqual(calls.count("test.zip.part003"), 1)
            self.assertEqual(calls.count("test.zip.part002"), 2)
            receipt = json.loads((root / "verified.json").read_text())
            self.assertEqual(receipt["parts"], manifest["parts"])

    def test_corrupt_remote_content_never_creates_cleanup_receipt(self):
        with tempfile.TemporaryDirectory(dir=archive.ROOT / "cache") as tmp:
            root = Path(tmp)
            manifest, data = self.fixture(root)

            def remote(tag, name):
                if name == "manifest.json":
                    return io.BytesIO(json.dumps(manifest).encode())
                return io.BytesIO(b"corrupt")

            with patch.object(verifier, "location", return_value=root), \
                 patch.object(verifier, "read_manifest", return_value=manifest), \
                 patch.object(verifier, "remote_part", side_effect=remote):
                with self.assertRaisesRegex(RuntimeError, "not verified"):
                    verifier.verify("test", 2)
            self.assertFalse((root / "verified.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
