"""Persist the exact observed quartet, without future-outcome labels."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3

import numpy as np

from experiment import ROOT, generate, quartet_data


def run(stage):
    output = ROOT / stage
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    start, end = manifest["data"]["train"]["ids"]
    base = generate(end - start + 1, start)
    assert base.fingerprint() == manifest["data"]["train"]["sha256"]
    data = quartet_data(base)
    path = output / "quartet_observations.sqlite"
    if path.exists():
        raise FileExistsError(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "CREATE TABLE episodes (id INTEGER, variant INTEGER, context BLOB, body BLOB,"
            "PRIMARY KEY(id, variant))"
        )
        connection.execute(
            "CREATE TABLE events (id INTEGER, variant INTEGER, time INTEGER, observation BLOB,"
            "PRIMARY KEY(id, variant, time))"
        )
        connection.executemany(
            "INSERT INTO episodes VALUES (?, ?, ?, ?)",
            ((int(identity), row // len(base), data.context[row].tobytes(), data.body[row].tobytes())
             for row, identity in enumerate(data.ids)),
        )
        connection.executemany(
            "INSERT INTO events VALUES (?, ?, ?, ?)",
            ((int(identity), row // len(base), step, data.images[row, step].tobytes())
             for row, identity in enumerate(data.ids)
             for step in range(data.images.shape[1])),
        )
        connection.commit()
        count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == len(data) * data.images.shape[1]
        row = connection.execute(
            "SELECT observation FROM events WHERE id=? AND variant=1 AND time=?",
            (int(base.ids[0]), int(base.cue_times[0, 1])),
        ).fetchone()
        np.testing.assert_array_equal(
            np.frombuffer(row[0], np.float32),
            data.images[len(base), base.cue_times[0, 1]].ravel(),
        )
    report = {
        "stage": stage, "histories": len(base), "variants": 4,
        "records": count, "size_bytes": path.stat().st_size,
        "quartet_fingerprint": data.fingerprint(),
        "labels_stored": False,
    }
    (output / "quartet_archive_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    for stage in ("development", "confirmation"):
        run(stage)
