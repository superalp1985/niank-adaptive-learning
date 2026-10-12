# V12 audit implementation record

Training code, configuration, protocol and inherited simulator sources were
hashed before the run. None was edited during practice. Audit and report
scripts were added separately; they do not change learning or probe behavior.

## SQLite audit access

The first audit used a query for each operation window:

```sql
SELECT id,world,image FROM events WHERE time=? ORDER BY world;
```

`EXPLAIN QUERY PLAN` reported `SCAN events` and
`USE TEMP B-TREE FOR ORDER BY`. The table has an integer primary-key ID,
but no index on the time column. The completed `state_content` audit is
retained in `pilot/AUDIT_state_content.json`, with its input hashes and
audit-source hash. Its exact source is preserved as `audit_fullscan_snapshot.py`.
The process exited successfully. Training was neither interrupted nor repeated.

The subsequent audits read once in primary-key order and consume one batch
per time window:

```sql
SELECT id,world,image,time FROM events ORDER BY id;
```

Every row still has to satisfy `id = step * worlds + world`, world ordering,
and event-time equality. Physics, rewards, directory reconstruction, legal
choices and diagnostics use the same formulas and assertions. The archived
source files make this access-only change reviewable.

The final audit rehashes every partial audit's raw inputs and checks its
source hash against these preserved implementations before aggregating.
`AUDIT.json` records the implementation hash used for each mode.
Use `audit.py`, rather than the preserved first version, for later checking.

## Scope

All generated data, logs, caches and temporary archive files stay on E:.
Six boundary contracts passed before training and in the final source state.
These are engineering checks, not repetitions of completed scientific runs.
Backpropagation remains internal to each local module. No end-to-end
gradient connects numerical prediction, group selection and retrieval plans.
