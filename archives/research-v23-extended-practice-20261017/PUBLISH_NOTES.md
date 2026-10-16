# V23 Publication

2026-10-17. Practice and all frozen observations finished successfully.
Completed configurations were not rerun, and running algorithm source was
not changed during analysis. Code and reports were published at main commit
`d73b4a2fd86932ce367fde6a3f1bed006253e156`.

The immutable snapshot contains 595 files, 1,356,003,316 source bytes and
15 ZIP parts totaling 599,596,803 bytes. ZIP SHA-256:
`d94aa5ddd76a93ba8e12bfa73947a248af47b975c82ec7b310048cb53fc24192`.

The disposable upload repository has HEAD
`455a49617065f523cd221f5bc7d4726e44305764`. GitHub workflow
`38966723767` completed successfully with that HEAD. Release `415378631`
has 16 assets: all 15 parts and manifest.json. Asset names, sizes and
SHA-256 metadata matched the local manifest; see release.json.

All 15 parts were subsequently read back in full and passed length and
SHA-256 checks. The remote manifest matched the local manifest. Verification
streamed data without saving download copies on C; see verified.json.

The default cleanup plan was filtered to exactly 12 observation databases
under stage_probes, final_probe, diagnostics and exploring_frozen_probe.
Entries were checked against the immutable manifest and workspace boundary
before the verified cleanup tool ran.

Cleanup removed those 12 databases, freeing 391,086,080 bytes, and 31 verified
ZIP/part/disposable Git-blob files, freeing 1,795,400,186 bytes. Total reclaimed
space was 2,186,486,266 bytes, including temporary duplicate copies.
See cleanup.json and staging-cleanup.json.

Both main training databases, all 12 model milestones and the V22 continuation
source remain. All 583 retained snapshot files matched their hashes after
cleanup and before publication-status documents were updated. The V22 source
database hash was unchanged. Other versions, shared environments and caches
were preserved; see retained-check.json.

The temporary remote upload branch was deleted. The Release and immutable
snapshot remain available. Post-publication status updates and receipts are
stored on main.
