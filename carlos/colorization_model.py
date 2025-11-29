from huggingface_hub import hf_hub_download

# Downloader script
def colorization_execute():
    REPO_ID = "ayushshah/imagecolorization"
    FILENAME = "model.py"

    hf_hub_download(
        repo_id="ayushshah/imagecolorization",
        filename="model.py",
        local_dir=".",
        #local_dir_use_symlinks=False
    )

    weights_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="model.safetensors"
    )

    return weights_path
