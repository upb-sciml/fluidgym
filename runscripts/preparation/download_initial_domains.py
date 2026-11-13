"""Download the initial-domain dataset from HuggingFace into the local data path.

Usage:
    python runscripts/preparation/download_initial_domains.py
"""

from huggingface_hub import snapshot_download

import fluidgym

fluidgym.config.update("local_data_path", "./local_data")

snapshot_download(
    repo_id="upb-sciml/fluidgym-data",
    repo_type="dataset",
    local_dir=fluidgym.config.local_data_path,
    ignore_patterns=["*.png"],
)
