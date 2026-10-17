# V24 Publication

2026-10-17. V24 finite practice, audits, frozen observations and analysis
finished successfully. Completed configurations were not rerun. Code and
report were published at main commit `c3cfff0366137f172dcd9c1709384d92cbd740dd`.

The immutable snapshot contains 539 files and 2,424,990,243 source bytes.
Its ZIP is 1,168,280,006 bytes with SHA-256
`5b6c65fde4a3547aa26b3d6c1dc08b9c0b0d014efabdaf8189e7e054dc25cf03`,
split into 28 ZIP parts.

The disposable upload repository had HEAD
`27faa815f6ba2163de3aba861641bed720c5b23d`. GitHub workflow
`39001519883` completed successfully with that HEAD. Release `415575715`
has 29 assets: all 28 parts and manifest.json. Asset names, sizes and
SHA-256 metadata matched the local manifest; see release.json.

All 28 parts were then streamed from the Release and read in full. Every
length and SHA-256 matched, and the remote manifest matched the local
manifest. No download copy was retained; see verified.json.

The cleanup plan was filtered to exactly 11 frozen observation databases
under stage_probes, final_probe and diagnostics. Every selected path and
hash was checked against the immutable manifest before deletion.

Cleanup removed those 11 databases, freeing 385,236,992 bytes. Staging
cleanup then removed 57 verified ZIP/part/disposable Git-blob files, freeing
3,499,727,084 bytes. Total reclaimed space was 3,884,964,076 bytes, including
temporary duplicate copies. See cleanup.json and staging-cleanup.json.

Both main training databases, all 12 model milestones/final models and the
V23 continuation source remain. All 528 retained snapshot files matched their
manifest hashes after cleanup. The V23 source database remained SHA-256
`22341d95502d47382f63a276d33aa2a21c12da09b22842a3e1900215335642fe`.
See retained-check.json.

The temporary archive branch was deleted. The Release and immutable snapshot
remain available. Post-publication status updates and receipts are stored on
main.
