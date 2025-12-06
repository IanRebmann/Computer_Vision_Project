import gradio as gr
from PIL import Image
import torch
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Model IDs / paths ----
PRETRAINED_ID = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"          # crystal clear
FINETUNED_ID = "swin2sr_div2k_finetuned_x4_1000steps"                     # smooth (local folder in repo)

# ---- Load processors ----
processor_pre = AutoImageProcessor.from_pretrained(PRETRAINED_ID)
processor_ft  = AutoImageProcessor.from_pretrained(FINETUNED_ID, local_files_only=True)

# ---- Load models ----
model_pre = Swin2SRForImageSuperResolution.from_pretrained(PRETRAINED_ID).to(device)
model_ft  = Swin2SRForImageSuperResolution.from_pretrained(FINETUNED_ID, local_files_only=True).to(device)

model_pre.eval()
model_ft.eval()

# ---- Inference function ----
def swin2sr_upscale(input_image: Image.Image, mode: str):
    """
    Run 4x super-resolution using Swin2SR.
    mode: "Crystal clear (pretrained)" or "Smooth (fine-tuned)".
    """
    if input_image is None:
        return None

    if mode == "Smooth (fine-tuned)":
        model = model_ft
        processor = processor_ft
    else:
        model = model_pre
        processor = processor_pre

    inputs = processor(images=input_image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    sr_tensor = outputs.reconstruction.squeeze().clamp(0, 1)
    sr_array = (sr_tensor.mul(255).byte().cpu().permute(1, 2, 0).numpy())
    sr_image = Image.fromarray(sr_array)

    return sr_image

# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("# Image Super-Resolution (Swin2SR x4)")
    gr.Markdown(
        "Choose **Crystal clear (pretrained)** for the original Swin2SR model, "
        "or **Smooth (fine-tuned)** for the Swin2SR version we fine-tuned on DIV2K patches."
    )

    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload low-res image")
        output_image = gr.Image(type="pil", label="4x Super-resolved image")

    mode_dropdown = gr.Dropdown(
        label="Style",
        choices=["Crystal clear (pretrained)", "Smooth (fine-tuned)"],
        value="Crystal clear (pretrained)",
        interactive=True,
    )

    run_btn = gr.Button("Upscale")

    run_btn.click(
        fn=swin2sr_upscale,
        inputs=[input_image, mode_dropdown],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch()
