# V21 Publication

2026-10-16. Resumed the existing immutable snapshot without preparing it
again or rerunning training. Source commit:
`2db48652fae689827a264b1d3b1a3fae3d92461a`.

The snapshot contains 796 files, 12,728,322,751 source bytes and 205 ZIP
parts totaling 8,563,962,944 bytes. The zero-byte `run.lock` captured
in the original snapshot remains part of that immutable archive.

All 105 bounded push batches completed. A connection reset after batch 99
was handled by checking the remote HEAD and resuming the remaining batches.
The disposable upload repository uses the existing local HTTP proxy and
disables delta search; the main and global Git configurations are unchanged.

The uploaded release-trigger commit is
`426d310062e614c088d3922ee3752e18a8022b84`.
GitHub workflow `38895949762` completed successfully at that exact commit.
Release `414912128` contains 206 uploaded assets: 205 parts and
`manifest.json`. Paginated API checks confirmed every asset's name, size,
state and SHA-256 metadata. See `release.json`.

The workflow also validated the concatenated ZIP SHA-256:
`e3cfa131c72d5d0567fed7fa51ad0e638e8b38b6577bb0805892c6076f87299a`.

All 205 parts were subsequently read back in full from the Release and
passed size and SHA-256 checks. The remote manifest matched the local
file exactly. Readback started with four workers, then resumed with eight
workers using only completed hash receipts. No downloaded archive was
saved on C. See `verified.json`.

The cleanup plan excluded all caches and retained four small final
`model.pt` files plus
`tracking_v21/continued_practice/reversible/events.sqlite`.
Every deletion candidate was checked against the immutable manifest,
restricted to V21, and hashed again before deletion.

Cleanup removed 639 archived binary files, freeing 8,643,807,359 bytes,
and 411 disposable ZIP/part/Git-blob files, freeing another
25,683,591,617 bytes. Total reclaimed: 34,327,398,976 bytes (34.33 GB),
including duplicate staging copies. No changed files were deleted.
Only 674,258 bytes of staging metadata remain.

All 156 retained snapshot files matched their original hashes after
cleanup and before updating the status document. The previously removed
zero-byte `run.lock` is recorded separately; its archived copy is intact.
See `retention-check.json`, `cleanup.json` and `staging-cleanup.json`.
Other experiments, project caches, shared environments and original
models were preserved.

The temporary remote upload branch was deleted after full verification.
Release `414912128` and its immutable assets remain available.
Publication status and receipts are stored on main.
