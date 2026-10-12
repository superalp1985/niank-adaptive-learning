"""Release-side validation. No credentials are read or printed."""

import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
assert manifest["repository"] == "superalp1985/niank-adaptive-learning"
assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,90}", manifest["tag"])
whole = hashlib.sha256()
for part in manifest["parts"]:
    path = (root / part["name"]).resolve()
    assert path.parent == root
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
            whole.update(block)
    assert path.stat().st_size == part["bytes"]
    assert sha.hexdigest() == part["sha256"]
assert whole.hexdigest() == manifest["zip_sha256"]
print(json.dumps({"validated_parts": len(manifest["parts"]), "zip_sha256": whole.hexdigest()}))
