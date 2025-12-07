# src/pipelines/colorize/hf_weights.py
from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "ayushshah/imagecolorization"
DEFAULT_FILENAME = "model.safetensors"

def download_colorization_weights(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
):
    """
    Returns local path to safetensors weights.
    """
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return weights_path
