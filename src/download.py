#!/usr/bin/env python3

from huggingface_hub import snapshot_download
model_id="LeoLM/leo-hessianai-7b"
snapshot_download(repo_id=model_id, local_dir="models/leolm-7b",
                  local_dir_use_symlinks=False, revision="main")
