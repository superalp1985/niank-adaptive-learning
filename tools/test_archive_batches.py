"""Local-Git integration tests for resumable immutable archive publishing."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import archive_snapshot as archive


class BatchContracts(unittest.TestCase):
    def test_large_archive_final_commit_triggers_upload_path_filter(self):
        with tempfile.TemporaryDirectory(dir=archive.ROOT / "cache") as tmp:
            root = Path(tmp)
            stage = root / ".archive-staging/test"
            stage.mkdir(parents=True)
            repo, remote, tree, manifest = self.fixture(stage)
            manifest.update(zip_bytes=2000 * 1024 * 1024, zip_sha256="a" * 64)
            with patch.object(archive, "ROOT", root), \
                 patch.object(archive, "location", return_value=root / "ledger"), \
                 patch.object(archive, "read_manifest", return_value=manifest):
                archive.publish("test")
            changed = archive.git(["diff", "--name-only", "HEAD^", "HEAD"], repo)
            self.assertEqual(changed, "upload/release-ready.json")
            self.assertIn(".github/workflows", archive.git(["ls-tree", "-r", "HEAD"], repo))

    def fixture(self, root):
        repo, remote = root / "upload-repo", root / "bare-remote"
        repo.mkdir()
        archive.git(["init", "--bare", str(remote)], root)
        archive.git(["init", "-b", "codex/archive-test"], repo)
        archive.git(["remote", "add", "origin", str(remote)], repo)
        upload = repo / "upload"
        upload.mkdir()
        archive.dump(upload / "manifest.json", dict(tag="test"))
        workflow = repo / ".github/workflows/publish.yml"
        workflow.parent.mkdir(parents=True)
        workflow.write_text("name: test\n", encoding="utf-8")
        for i in range(3):
            (upload / f"test.zip.part{i + 1:03d}").write_bytes(bytes([i]) * 128)
        archive.git(["add", "--all"], repo)
        archive.git(["-c", "user.name=Archive Test", "-c", "user.email=test@example.invalid",
                     "commit", "-qm", "Immutable snapshot"], repo)
        archive.git(["config", "user.name", "Archive Test"], repo)
        archive.git(["config", "user.email", "test@example.invalid"], repo)
        tree = archive.git(["rev-parse", "HEAD^{tree}"], repo)
        return repo, remote, tree, dict(parts=[{}, {}, {}])

    def test_published_final_tree_is_identical_and_workflow_is_last(self):
        with tempfile.TemporaryDirectory(dir=archive.ROOT / "cache") as tmp:
            root = Path(tmp)
            repo, remote, tree, manifest = self.fixture(root)
            with patch.object(archive, "location", return_value=root / "ledger"):
                archive.publish_batched("test", repo, "codex/archive-test", manifest)
            final = archive.git(["rev-parse", "refs/heads/codex/archive-test"], remote)
            self.assertEqual(archive.git(["rev-parse", f"{final}^{{tree}}"], remote), tree)
            prior = archive.git(["ls-tree", "-r", f"{final}^"], remote)
            self.assertNotIn(".github/workflows", prior)
            ledger = json.loads((root / "ledger/local-publish-batches.json").read_text())
            self.assertEqual(ledger["parts_per_push"], 2)
            self.assertEqual(len(ledger["commits"]), 4)

    def test_network_failure_resumes_without_rewriting_published_history(self):
        with tempfile.TemporaryDirectory(dir=archive.ROOT / "cache") as tmp:
            root = Path(tmp)
            repo, remote, tree, manifest = self.fixture(root)
            original_git, calls = archive.git, []

            def fail_third_push(args, cwd):
                if args[0] == "push":
                    calls.append(args)
                    if len(calls) == 3:
                        raise RuntimeError("simulated connection reset")
                return original_git(args, cwd)

            with patch.object(archive, "location", return_value=root / "ledger"):
                with patch.object(archive, "git", side_effect=fail_third_push):
                    with self.assertRaisesRegex(RuntimeError, "simulated connection"):
                        archive.publish_batched("test", repo, "codex/archive-test", manifest)
                published = original_git(["rev-parse", "refs/heads/codex/archive-test"], remote)
                archive.publish_batched("test", repo, "codex/archive-test", manifest)
            final = original_git(["rev-parse", "refs/heads/codex/archive-test"], remote)
            self.assertEqual(original_git(["merge-base", published, final], remote), published)
            self.assertEqual(original_git(["rev-parse", f"{final}^{{tree}}"], remote), tree)


if __name__ == "__main__":
    unittest.main(verbosity=2)
