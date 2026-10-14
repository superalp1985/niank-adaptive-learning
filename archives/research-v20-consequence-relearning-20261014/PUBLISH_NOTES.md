# V20 Publication

2026-10-15. The snapshot was not regenerated and no training was rerun.
Code and reports were published at main commit
`7d69e4ae10b26ad90b426125655cbfa662765993`.

The immutable snapshot contains 345 files, 5,228,678,023 source bytes and
85 ZIP parts totaling 3,562,867,242 bytes. ZIP SHA-256:
`47e244fba308a5aacc0bf008d835956134810e2ca4b0cdbd0afef8e3db03de98`.

All 45 bounded push batches completed. Delta search was disabled only in the
publisher subprocess environment; the main repository and global Git
configuration were not changed. The disposable upload repository has HEAD
`c203599faea6f8eab88d47f33492e49d2e06405c`.
GitHub workflow `38599676072` completed successfully with that same HEAD.
Release `412954345` has 86 uploaded assets: 85 parts and `manifest.json`.
Their names, sizes and SHA-256 metadata match the local manifest.
See `release.json`.

One local API check initially counted an extra PowerShell array wrapper instead
of its 86 assets. Unwrapping the response resolved that check without
republishing, changing assets or changing the manifest.

Full remote readback and any subsequent cleanup are recorded separately in
`verified.json`, `cleanup.json` and `staging-cleanup.json`. Publication alone
does not authorize deletion.

All 85 parts were subsequently read back in full from the Release and passed
length and SHA-256 checks. The remote manifest hash matches the local file.
No downloaded archive was saved on C.

The cleanup plan excluded all project caches and was independently checked
against the immutable manifest. Cleanup removed 277 archived binary files,
freeing 5,224,755,710 bytes, and 171 disposable ZIP/part/Git-blob files,
freeing another 10,686,815,620 bytes. The total is 15,911,571,330 bytes,
including duplicate staging copies.

Three final `model.pt` files and all 65 small snapshot files were retained.
All 68 retained files matched their snapshot hashes after cleanup and before
the publication-status documents were updated. All caches, ancestor models,
other experiments and shared environments were preserved. No packed objects
were present in the disposable upload repository.

The temporary remote upload branch was deleted; Release `412954345` remains
available with all 86 assets. The Release snapshot remains immutable.
Post-publication status updates and all receipts are stored on main.
