"""Upload the local initial-domain data to the HuggingFace dataset.

Usage:
    python runscripts/preparation/upload_initial_domains.py
"""

from huggingface_hub import HfApi

import fluidgym

fluidgym.config.update("local_data_path", "./local_data")

api = HfApi()

api.upload_large_folder(
    folder_path=fluidgym.config.local_data_path,
    repo_id="upb-sciml/fluidgym-data",
    repo_type="dataset",
    ignore_patterns=["*.png"],
)
