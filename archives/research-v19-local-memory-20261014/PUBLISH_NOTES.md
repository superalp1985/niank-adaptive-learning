# V19 publication note

2026-10-14. Snapshot content and part hashes were not changed during publication.

The first push spent over six minutes in `git pack-objects`, with approximately
1 GB working memory and no bulk output from that process. Its identity and parent
chain were checked, then that pack process alone was stopped. The publishing
wrapper exited with an error; all original data and archive parts were retained.

Only the disposable upload repository was then configured with
`git config --local pack.window 0`. The same immutable snapshot was published
again, without rerunning training, regenerating the ZIP, or modifying any source.
On the next push, the pack process emitted over 203 MB while using much less CPU
time. This supports delta search as the observed local packing bottleneck.

For later snapshots containing already compressed parts, consider disabling
delta search in the disposable upload repository before its first push.
Do not apply this setting to the main research repository or change the data.
Successful publication, full remote verification and cleanup require the
separate receipts in this directory; this note is not a verification receipt.

The second push succeeded. GitHub run `38496067889` completed successfully with
HEAD `954c444671a243059a8f61a2cf9749187956f611`, matching the upload repository.
All 20 release parts were subsequently read back in full and passed length and
SHA-256 checks. Parts 001 and 016 initially returned incomplete streams; these
were rejected and their complete retries passed. See `verified.json`.

After verification, the temporary remote upload branch was deleted. The Release
remains available. Cleanup receipts record 1,047,011,820 bytes removed from
verified research data/project caches and 2,487,457,786 bytes from staging.
Three small `model.pt` working checkpoints were retained.
