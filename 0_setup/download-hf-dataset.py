from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/PE-Video",
    repo_type="dataset",
    local_dir=".cache/huggingface/datasets/facebook/PE-Video",
    local_dir_use_symlinks=False,
    allow_patterns=["train/*", "test/*"],
    ignore_patterns="extended/*"
)