import gradio as gr
from PIL import Image
import torch
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

# ---- Load model and processor ----

MODEL_ID = "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr"

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = Swin2SRForImageSuperResolution.from_pretrained(MODEL_ID)
model.eval()


# ---- Inference function ----

def swin2sr_upscale(input_image: Image.Image):
    """
    Run 4x super-resolution using Swin2SR.

    Returns the super-resolved image.
    """
    if input_image is None:
        return None

    # Prepare inputs for the model
    inputs = processor(images=input_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process back to PIL image
    # For Swin2SR, the SR image is in outputs.reconstruction
    sr_tensor = outputs.reconstruction.squeeze().clamp(0, 1)
    sr_array = (sr_tensor.mul(255).byte().cpu().permute(1, 2, 0).numpy())
    sr_image = Image.fromarray(sr_array)

    return sr_image


# ---- Gradio UI ----

demo = gr.Interface(
    fn=swin2sr_upscale,
    inputs=gr.Image(type="pil", label="Upload low-res image"),
    outputs=gr.Image(type="pil", label="4x Super-resolved image (Swin2SR)"),
    title="Image Super-Resolution (Swin2SR x4)",
    description=(
        "Real-world 4x super-resolution using the Swin2SR transformer "
        "model (caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr)."
    ),
)

if __name__ == "__main__":
    demo.launch()
