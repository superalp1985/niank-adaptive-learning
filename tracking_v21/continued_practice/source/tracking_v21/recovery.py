"""Explicit weights-only continuation, then transactionally checkpointed practice."""

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from tracking_v17.run import dump
from tracking_v20.run import Pending
from tracking_v21.directory import Directory
from tracking_v21.run import ROOT, REPO, Session, SettlingPending, probe


class DeferredCommit:
    def __init__(self, connection):
        self.connection = connection

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def commit(self):
        pass


def make_durable(session):
    session.db.execute("CREATE TABLE IF NOT EXISTS runtime_checkpoint("
                       "id INTEGER PRIMARY KEY,step INTEGER,sha256 TEXT,payload BLOB)")
    session.db.commit()
    session.db = DeferredCommit(session.db)
    if isinstance(session.directory, Directory):
        session.directory.db = session.db


def checkpoint(session):
    state = {key: value for key, value in vars(session).items()
             if key not in ("db", "agent", "directory", "pending", "start")}
    agent = copy.copy(session.agent)
    agent.query = None
    directory = copy.copy(session.directory)
    if isinstance(directory, Directory):
        directory.db = None
    payload = dict(format=1, session=state, agent=agent, directory=directory,
        queues=session.pending.queues, elapsed=time.perf_counter() - session.start,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(), python_rng=random.getstate())
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    session.db.execute("INSERT OR REPLACE INTO runtime_checkpoint VALUES(1,?,?,?)",
                       (session.step, hashlib.sha256(data).hexdigest(), data))
    session.db.connection.commit()


def advance(session, steps, epochs):
    point = session.advance(steps, epochs)
    checkpoint(session)
    return point


def restore(path):
    path = Path(path)
    db = sqlite3.connect(path / "events.sqlite")
    row = db.execute("SELECT step,sha256,payload FROM runtime_checkpoint WHERE id=1").fetchone()
    if row is None or hashlib.sha256(row[2]).hexdigest() != row[1]:
        db.close()
        raise RuntimeError("Missing or corrupt committed runtime checkpoint")
    # Only locally created, hash-checked snapshots are loaded as Python objects.
    saved = torch.load(io.BytesIO(row[2]), weights_only=False)
    session = Session.__new__(Session)
    vars(session).update(saved["session"])
    session.output = path
    session.db = DeferredCommit(db)
    session.agent, session.directory = saved["agent"], saved["directory"]
    session.start = time.perf_counter() - saved["elapsed"]
    if isinstance(session.directory, Directory):
        session.directory.db = session.db
        session.pending = SettlingPending(session.worlds, session.directory, lambda: session.step)
        session.agent.query = lambda: session.directory.query(
            session.memory.full_features, session.proprio, session.step)
    else:
        session.pending = Pending(session.worlds)
    session.pending.queues = saved["queues"]
    if session.step != row[0]:
        raise RuntimeError("Runtime step differs from transaction")
    for file in path.glob("transitions_*.npz"):
        if int(file.stem.split("_")[-1]) >= session.chunk:
            destination = path / "uncommitted" / str(time.time_ns()) / file.name
            destination.parent.mkdir(parents=True)
            file.replace(destination)
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
    dump(path / "restored.json", dict(step=session.step, full_runtime=True))
    return session


def configuration():
    config = json.loads((ROOT / "config.json").read_text())
    config.update(stage="continued_practice", world_seed=831000,
                  milestones=[8192, 20480], ancestor_v21_steps=4096)
    return config


def observe(session, path, config, shift=None):
    summary = path / "summary.json"
    if summary.exists():
        return json.loads(summary.read_text())
    if path.exists():
        saved = path.parent / "interrupted_observations" / (
            path.name + "_" + str(time.time_ns()))
        saved.parent.mkdir(exist_ok=True)
        path.replace(saved)
    fixed = dict(config, world_seed=821000)
    return probe(session, path, fixed, shift=shift)


def worker(config, arm):
    path = ROOT / config["stage"] / arm
    if (path / "COMPLETE.json").exists():
        return
    torch.manual_seed(config["model_seed"])
    torch.cuda.manual_seed_all(config["model_seed"])
    if path.exists():
        session = restore(path)
    else:
        session = Session(path, arm, config["world_seed"], config)
        source = ROOT / "reversible_practice" / arm / "checkpoint_4096.pt"
        session.agent.load(torch.load(source, weights_only=True, map_location="cuda")["model"])
        make_durable(session)
        checkpoint(session)
    try:
        for milestone in config["milestones"]:
            while session.step < milestone:
                point = advance(session, config["rollout"], config["epochs"])
                if session.chunk % 8 == 0:
                    print(json.dumps(dict(event="progress", arm=arm, **point)), flush=True)
            result_path = path / f"result_{milestone}.json"
            if not result_path.exists():
                if session.step != milestone:
                    raise RuntimeError("A missing earlier observation cannot use later weights")
                torch.save(dict(model=session.agent.state(), step=session.step),
                           path / f"checkpoint_{milestone}.pt")
                result = session.summary()
                result["probe"] = observe(session, path / f"probe_{milestone}", config)
                dump(result_path, result)
                print(json.dumps(dict(event="milestone", arm=arm, step=milestone,
                    safe=result["probe"]["safe_contacts"],
                    collisions=result["probe"]["collision_rate"])), flush=True)
        for shift in (0, 1):
            result = observe(session, path / f"final_shift_{shift}", config, shift)
            dump(path / f"final_shift_{shift}.json", result)
        torch.save(dict(model=session.agent.state()), path / "model.pt")
        dump(path / "summary.json", session.summary())
        dump(path / "COMPLETE.json", dict(complete=True, steps=session.step,
            lineage_steps=4096 + session.step, weights_only_boundary=4096))
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("random", "reversible"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = configuration()
    if args.arm:
        worker(config, args.arm)
        return
    output = ROOT / config["stage"]
    output.mkdir(exist_ok=True)
    import msvcrt
    lock = (output / "run.lock").open("a+b")
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        frozen = ROOT / "reversible_practice" / "manifest.json"
        original = json.loads(frozen.read_text())
        sources = dict(original["sources"])
        for name in ("recovery.py", "RECOVERY_ZH.md", "test_recovery.py"):
            sources["tracking_v21/" + name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        manifest = output / "manifest.json"
        if not manifest.exists():
            for name, sha in sources.items():
                source = REPO / name
                if hashlib.sha256(source.read_bytes()).hexdigest() != sha:
                    raise RuntimeError("Frozen training source changed: " + name)
                destination = output / "source" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            ancestors = {arm: hashlib.sha256((ROOT / "reversible_practice" / arm /
                         "checkpoint_4096.pt").read_bytes()).hexdigest() for arm in config["arms"]}
            dump(manifest, dict(config=config, sources=sources, ancestors=ancestors,
                                continuation="weights_only_at_4096", gpu=torch.cuda.get_device_name()))
        else:
            existing = json.loads(manifest.read_text())
            if existing["config"] != config or existing["sources"] != sources:
                raise RuntimeError("Continuation configuration or sources changed")
        handles, children = [], []
        try:
            for arm in config["arms"]:
                handle = (output / f"{arm}.log").open("a", encoding="utf-8")
                handles.append(handle)
                children.append(subprocess.Popen(
                    [sys.executable, "-B", "-m", "tracking_v21.recovery", "--arm", arm],
                    cwd=REPO, stdout=handle, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW))
            dump(output / "RUNNING.json", dict(pid=os.getpid(), workers=[c.pid for c in children]))
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError("Worker failed; inspect logs before resuming")
                time.sleep(2)
            if any(child.returncode for child in children):
                raise RuntimeError("Worker failed")
            dump(output / "COMPLETE.json", dict(complete=True, config=config,
                continuation="weights_only_at_4096"))
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
            for handle in handles:
                handle.close()
    except BaseException:
        dump(output / "FAILED.json", dict(error=traceback.format_exc()))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
