"""Atomic small artifacts and validated database publication on the same volume."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".partial_"+uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_complete(path, validate):
    """An invalid marker is evidence, never proof that a run completed."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or not validate(result):
            raise ValueError("Incomplete completion marker")
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, UnicodeError):
        path.rename(path.with_name(path.name+".invalid_"+uuid.uuid4().hex))
        return None
    return result


def runtime_identity(connection):
    if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise RuntimeError("SQLite integrity check failed")
    row = connection.execute("SELECT step,sha256,payload FROM runtime WHERE id=1").fetchone()
    if row is None or hashlib.sha256(row[2]).hexdigest() != row[1]:
        raise RuntimeError("No complete hash-verified runtime")
    return row[0], row[1]


def validate_database(path):
    path = Path(path).resolve()
    connection = sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)
    try:
        return runtime_identity(connection)
    except sqlite3.DatabaseError as error:
        raise RuntimeError("Invalid runtime database: "+str(path)) from error
    finally:
        connection.close()


def backup_database(source, destination, progress=None):
    """Publish only a complete, validated backup; failed temporary files remain."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError("Database backup requires a distinct, absent destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+".partial_"+uuid.uuid4().hex)
    src = sqlite3.connect(source.as_uri()+"?mode=ro", uri=True)
    dst = None
    try:
        expected = runtime_identity(src)
        dst = sqlite3.connect(temporary)
        src.backup(dst, pages=256, progress=progress)
        if runtime_identity(dst) != expected:
            raise RuntimeError("Source runtime changed during database backup")
    finally:
        if dst is not None:
            dst.close()
        src.close()
    if destination.exists():
        raise FileExistsError(destination)
    # On Windows rename fails if a destination appeared, rather than replacing it.
    temporary.rename(destination)


def preserve_incomplete_bootstrap(output):
    """Recover only an unpublished bootstrap; never replace practiced state."""
    output = Path(output)
    database = output/"events.sqlite"
    if not database.exists():
        return
    try:
        validate_database(database)
        return
    except RuntimeError as error:
        if list(output.glob("learner*.pt")) or (output/"progress.json").exists():
            raise RuntimeError("Invalid practiced database preserved; refusing to rebootstrap") from error
    evidence = output/("interrupted_bootstrap_"+uuid.uuid4().hex)
    evidence.mkdir()
    for suffix in ("", "-wal", "-shm", "-journal"):
        file = output/("events.sqlite"+suffix)
        if file.exists():
            file.rename(evidence/file.name)
