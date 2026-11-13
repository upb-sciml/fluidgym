"""Upload the local `output` directory to the HuggingFace results dataset.

Usage:
    python runscripts/evaluation/upload_experimental_results.py
"""

from huggingface_hub import HfApi

import fluidgym

fluidgym.config.update("local_data_path", "./local_data")

api = HfApi()

api.upload_large_folder(
    folder_path="output",
    repo_id="upb-sciml/mhd_control",
    repo_type="dataset",
    ignore_patterns=[
        "**/.DS_Store",
        "*.png",
        "*.gif",
        "*.log",
        "*.pdf",
        "**/wandb/*",
        "hydra.yaml",
        "multirun.yaml",
        "overrides.yaml",
        "preparation/**",
        "validation/control_tol_check/**",
        "validation/control/**",
        "**/train_render/**",
        "**/train_videos/**",
        "**/.submitit/**",
        "**/tb/**",
        "**/42/**",
    ],
)
