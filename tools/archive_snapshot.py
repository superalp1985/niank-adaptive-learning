"""Archive, publish through ordinary Git, verify, and restore research snapshots."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "superalp1985/niank-adaptive-learning"
REMOTE = f"https://github.com/{REPOSITORY}.git"
CHUNK = 40 * 1024 * 1024
EXCLUDED = {".git", ".archive-staging", "__pycache__", "cache", "archives"}
BINARY = {".pt", ".pth", ".npz", ".npy", ".sqlite", ".sqlite3", ".db", ".log"}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def validate_tag(tag):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,90}", tag):
        raise ValueError("Invalid archive tag")
    return tag


def location(tag):
    return ROOT / "archives" / validate_tag(tag)


def read_manifest(tag):
    return json.loads((location(tag) / "manifest.json").read_text(encoding="utf-8"))


def git(arguments, cwd):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")
    result = subprocess.run(
        ["git", "-c", "credential.interactive=never", *arguments], cwd=cwd,
        env=env, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])
    return result.stdout.strip()


def prepare(tag, scope):
    target = location(tag)
    stage = ROOT / ".archive-staging" / tag
    if (target / "manifest.json").exists() or stage.exists():
        raise FileExistsError("Use a new tag; existing snapshots are immutable")
    scope_path = (ROOT / scope).resolve()
    if not scope_path.is_relative_to(ROOT):
        raise ValueError("Scope outside repository")
    stage.mkdir(parents=True)
    upload = stage / "upload-repo" / "upload"
    upload.mkdir(parents=True)
    (upload.parent / ".gitattributes").write_text("* -text\n", encoding="utf-8")
    paths = sorted(
        p for p in scope_path.rglob("*") if p.is_file()
        and not any(part in EXCLUDED or part.startswith(".venv") for part in p.relative_to(ROOT).parts)
    )
    if not paths:
        raise ValueError("Empty snapshot")
    entries = []
    archive = stage / "snapshot.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as output:
        for path in paths:
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                raise ValueError(f"Linked file refused: {path}")
            relative = path.relative_to(ROOT).as_posix()
            before = digest(path)
            output.write(path, relative)
            if digest(path) != before:
                raise RuntimeError(f"File changed while archiving: {relative}")
            entries.append(dict(path=relative, bytes=path.stat().st_size, sha256=before))
    with zipfile.ZipFile(archive) as check:
        if check.testzip() is not None:
            raise RuntimeError("Local ZIP integrity failure")
    parts = []
    with archive.open("rb") as stream:
        index = 1
        while block := stream.read(CHUNK):
            name = f"{tag}.zip.part{index:03d}"
            (upload / name).write_bytes(block)
            parts.append(dict(name=name, bytes=len(block), sha256=hashlib.sha256(block).hexdigest()))
            index += 1
    manifest = dict(
        format=1, repository=REPOSITORY, tag=tag, scope=scope,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_commit=git(["rev-parse", "HEAD"], ROOT),
        files=entries, source_bytes=sum(p["bytes"] for p in entries),
        zip_bytes=archive.stat().st_size, zip_sha256=digest(archive), parts=parts,
    )
    dump(target / "manifest.json", manifest)
    dump(upload / "manifest.json", manifest)
    (upload / "release-notes.md").write_text(
        f"# {tag}\n\nComplete research snapshot: {len(entries)} files, "
        f"{manifest['source_bytes']:,} source bytes.\n\n"
        "Includes successful and failed runs, raw observations, trajectories, "
        "checkpoints, source snapshots and reports. Part SHA-256 hashes and "
        "per-file hashes are in manifest.json. Restore with tools/archive_snapshot.py.\n",
        encoding="utf-8",
    )
    verifier = ROOT / "tools" / "verify_parts.py"
    shutil.copyfile(verifier, upload / "verify_parts.py")
    workflow = stage / "upload-repo" / ".github" / "workflows" / "publish-archive.yml"
    workflow.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / ".github" / "workflows" / "publish-archive.yml", workflow)
    print(json.dumps({"prepared": tag, "files": len(entries), "source_bytes": manifest["source_bytes"],
                      "zip_bytes": manifest["zip_bytes"], "parts": len(parts)}), flush=True)


def publish(tag):
    manifest = read_manifest(tag)
    repo = ROOT / ".archive-staging" / tag / "upload-repo"
    branch = f"codex/archive-{tag}"
    if not (repo / ".git").exists():
        git(["init", "-b", branch], repo)
        git(["remote", "add", "origin", REMOTE], repo)
        git(["config", "user.name", git(["config", "user.name"], ROOT)], repo)
        git(["config", "user.email", git(["config", "user.email"], ROOT)], repo)
        git(["add", "--all"], repo)
        git(["commit", "-qm", f"Publish immutable research archive {tag}"], repo)
    git(["push", "-u", "origin", branch], repo)
    print(json.dumps({"pushed_archive_branch": branch, "parts": len(manifest["parts"])}), flush=True)


def remote_part(tag, name):
    url = f"https://github.com/{REPOSITORY}/releases/download/{tag}/{name}"
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "niank-archive/1"}), timeout=120)


def verify(tag):
    manifest = read_manifest(tag)
    with remote_part(tag, "manifest.json") as source:
        remote_manifest = source.read()
    if json.loads(remote_manifest) != manifest:
        raise RuntimeError("Remote manifest content differs from local manifest")
    verified = []
    for part in manifest["parts"]:
        sha, size = hashlib.sha256(), 0
        with remote_part(tag, part["name"]) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
                size += len(block)
        if size != part["bytes"] or sha.hexdigest() != part["sha256"]:
            raise RuntimeError(f"Remote verification failed: {part['name']}")
        verified.append(dict(name=part["name"], bytes=size, sha256=sha.hexdigest()))
        print(json.dumps({"verified_part": part["name"], "bytes": size}), flush=True)
    dump(location(tag) / "verified.json", dict(
        repository=REPOSITORY, tag=tag, manifest_sha256=digest(location(tag) / "manifest.json"),
        remote_manifest_sha256=hashlib.sha256(remote_manifest).hexdigest(),
        verified_utc=dt.datetime.now(dt.timezone.utc).isoformat(), parts=verified,
    ))


def cleanup_plan(tag):
    manifest = read_manifest(tag)
    receipt = json.loads((location(tag) / "verified.json").read_text())
    if receipt["manifest_sha256"] != digest(location(tag) / "manifest.json") or receipt["parts"] != manifest["parts"]:
        raise RuntimeError("A complete matching remote verification receipt is required")
    entries, retained = [], []
    # Keep only the most recent V10 warm-start set; future scopes retain their final model set.
    for entry in manifest["files"]:
        path = Path(entry["path"])
        if path.suffix not in BINARY:
            continue
        keep = (entry["path"].startswith("allocation_v10/continued_practice/")
                and path.name == "model.pt")
        if manifest["scope"] != "." and path.name in ("model.pt", "resume.pt"):
            keep = True
        (retained if keep else entries).append(entry)
    # Generated caches are dispensable and are kept out of the uploaded research data.
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if (path.is_file() and not any(p in (".git", ".archive-staging") for p in relative.parts)
                and any(p in ("cache", "__pycache__") for p in relative.parts)):
            entries.append(dict(path=relative.as_posix(), bytes=path.stat().st_size, sha256=digest(path), cache=True))
    dump(location(tag) / "cleanup-plan.json", dict(
        root=str(ROOT.resolve()), tag=tag, manifest_sha256=receipt["manifest_sha256"],
        entries=entries, retained=retained,
    ))
    print(json.dumps({"cleanup_candidates": len(entries), "bytes": sum(x["bytes"] for x in entries),
                      "retained_binary_files": len(retained)}), flush=True)


def restore(tag, destination):
    manifest = read_manifest(tag)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f".{tag}.restore.zip"
    with archive.open("xb") as output:
        for part in manifest["parts"]:
            sha, size = hashlib.sha256(), 0
            with remote_part(tag, part["name"]) as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    sha.update(block)
                    size += len(block)
                    output.write(block)
            if size != part["bytes"] or sha.hexdigest() != part["sha256"]:
                raise RuntimeError(f"Bad remote part {part['name']}")
    if digest(archive) != manifest["zip_sha256"]:
        raise RuntimeError("ZIP SHA-256 mismatch")
    with zipfile.ZipFile(archive) as source:
        if set(source.namelist()) != {e["path"] for e in manifest["files"]}:
            raise RuntimeError("Unexpected ZIP members")
        for entry in manifest["files"]:
            path = (destination / entry["path"]).resolve()
            if not path.is_relative_to(destination):
                raise RuntimeError("Unsafe archive member")
            if path.exists():
                if path.is_file() and digest(path) == entry["sha256"]:
                    continue
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.open(entry["path"]) as data, path.open("xb") as output:
                shutil.copyfileobj(data, output, 1024 * 1024)
            if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                raise RuntimeError(f"Restored file mismatch: {entry['path']}")
    archive.unlink()
    print(json.dumps({"restored": len(manifest["files"]), "destination": str(destination)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "publish", "verify", "cleanup-plan", "restore"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--scope", default=".")
    parser.add_argument("--destination")
    args = parser.parse_args()
    validate_tag(args.tag)
    if args.operation == "prepare":
        prepare(args.tag, args.scope)
    elif args.operation == "publish":
        publish(args.tag)
    elif args.operation == "verify":
        verify(args.tag)
    elif args.operation == "cleanup-plan":
        cleanup_plan(args.tag)
    elif args.destination:
        restore(args.tag, args.destination)
    else:
        parser.error("restore requires --destination")
