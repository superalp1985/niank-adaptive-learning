"""Post hoc executed-episode diagnostics, never fed back into training."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    output = {}
    for mode in ("more_practice", "safe_contact"):
        episodes = {}
        folder = ROOT / "pilot" / mode / "probe_8192" / "heldout"
        for path in sorted(folder.glob("transitions_*.npz")):
            with np.load(path) as data:
                for step in range(len(data["done"])):
                    for world in range(data["done"].shape[1]):
                        key = (world, int(data["episode"][step, world]))
                        episode = episodes.setdefault(key, dict(steps=0, collision_steps=0,
                            success=False, impact=False, done=False, stopped_steps=0,
                            post_collision_steps=0, post_collision_stop=0))
                        episode["steps"] += 1
                        episode["collision_steps"] += int(data["collision"][step, world])
                        stopped = int(data["actions"][step, world] == 0)
                        episode["stopped_steps"] += stopped
                        if episode["collision_steps"]:
                            episode["post_collision_steps"] += 1
                            episode["post_collision_stop"] += stopped
                        episode["success"] |= bool(data["success"][step, world])
                        episode["impact"] |= bool(data["impact"][step, world])
                        episode["done"] |= bool(data["done"][step, world])
        finished = [e for e in episodes.values() if e["done"]]
        failed_collided = [e for e in finished if not e["success"] and e["collision_steps"]]
        output[mode] = dict(
            finished_episodes=len(finished),
            collided_failed_episodes=len(failed_collided),
            mean_failed_collided_duration=float(np.mean([e["steps"] for e in failed_collided])),
            mean_failed_collided_contact_steps=float(np.mean(
                [e["collision_steps"] for e in failed_collided])),
            post_collision_stop_fraction=sum(e["post_collision_stop"] for e in finished) /
                max(sum(e["post_collision_steps"] for e in finished), 1),
        )
    (ROOT / "pilot" / "POSTHOC_DIAGNOSTICS.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
