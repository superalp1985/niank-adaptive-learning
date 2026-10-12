# Third-party research resources

The original fly-connectome source is `eonsystemspbc/fly-brain`, pinned to
commit `a3db62f9436074e485c0278290c2164ed6150808`.
Its manifest is preserved in `DOWNLOAD_MANIFEST.json`.

The upstream simulator identifies its license as GPL-2.0-or-later.
The upstream `code/paper-phil-drosophila` directory has its own license.
Upstream source, datasets, language-model weights, and CUDA environments remain
separate dependencies; this repository does not relicense them.

Reduced connectome exports are derived from the upstream connectivity data;
their provenance is recorded in the original preparation scripts and manifests.
This repository does not claim that connection-count weights are pretrained
language or task knowledge.

The V5-V10 physical learning experiments use locally designed reduced tasks.
Claims and limitations for each experiment are in its protocol and report.
