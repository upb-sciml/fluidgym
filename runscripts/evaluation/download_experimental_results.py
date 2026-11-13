"""Download the HuggingFace results dataset into the local `output` directory.

Usage:
    python runscripts/evaluation/download_experimental_results.py
"""

from huggingface_hub import snapshot_download

import fluidgym

fluidgym.config.update("local_data_path", "./local_data")

snapshot_download(
    local_dir="output",
    repo_id="upb-sciml/mhd_control",
    repo_type="dataset",
)

