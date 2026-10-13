"""Stream independent release parts concurrently with resumable hash receipts."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import json
import time

from tools.archive_snapshot import dump, digest, location, read_manifest, remote_part


def verify(tag, workers=4):
    manifest = read_manifest(tag)
    with remote_part(tag, "manifest.json") as stream:
        remote = stream.read()
    if json.loads(remote) != manifest:
        raise RuntimeError("Remote manifest differs")
    manifest_hash = digest(location(tag) / "manifest.json")
    progress = location(tag) / "local-verify-progress.json"
    completed = {}
    if progress.exists():
        saved = json.loads(progress.read_text(encoding="utf-8"))
        if saved["manifest_sha256"] == manifest_hash:
            completed = {item["name"]: item for item in saved["parts"]}

    def check(part):
        for attempt in range(3):
            try:
                sha, size = hashlib.sha256(), 0
                with remote_part(tag, part["name"]) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        sha.update(block)
                        size += len(block)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
                continue
            result = dict(name=part["name"], bytes=size, sha256=sha.hexdigest())
            if result != part:
                print(json.dumps(dict(retry_mismatch=part["name"], attempt=attempt + 1,
                                      received_bytes=size, expected_bytes=part["bytes"])), flush=True)
                if attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                raise RuntimeError(f"Bad remote part after full readbacks: {part['name']}")
            return result

    pending = [p for p in manifest["parts"] if completed.get(p["name"]) != p]
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(check, part): part for part in pending}
        for job in as_completed(jobs):
            try:
                result = job.result()
            except Exception as error:
                failures.append((jobs[job]["name"], repr(error)))
                print(json.dumps(dict(unverified_part=jobs[job]["name"],
                                      error=repr(error))), flush=True)
                continue
            completed[result["name"]] = result
            dump(progress, dict(manifest_sha256=manifest_hash,
                                parts=list(completed.values())))
            print(json.dumps(dict(verified=result["name"], completed=len(completed),
                                  total=len(manifest["parts"]))), flush=True)
    if failures:
        raise RuntimeError(f"Some release parts were not verified: {failures}")
    receipt = dict(repository=manifest["repository"], tag=tag,
        manifest_sha256=manifest_hash,
        remote_manifest_sha256=hashlib.sha256(remote).hexdigest(),
        verified_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        parts=[completed[p["name"]] for p in manifest["parts"]])
    dump(location(tag) / "verified.json", receipt)
    print(json.dumps(dict(complete=True, parts=len(receipt["parts"]))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    arguments = parser.parse_args()
    verify(arguments.tag, arguments.workers)
