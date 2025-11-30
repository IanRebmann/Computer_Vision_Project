import gradio as gr
from PIL import Image
from super_image import HanModel, ImageLoader

model = HanModel.from_pretrained("eugenesiow/han", scale=4)

def han_sr(image: Image.Image):
    inputs = ImageLoader.load_image(image)
    preds = model(inputs)
    sr_path = "temp_sr.png"
    ImageLoader.save_image(preds, sr_path)
    return Image.open(sr_path).convert("RGB")

iface = gr.Interface(
    fn=han_sr,
    inputs=gr.Image(type="pil", label="Upload low-res image"),
    outputs=gr.Image(type="pil", label="4x HAN super-res image"),
    title="HAN Super-Resolution Demo",
)

if __name__ == "__main__":
    iface.launch()






'''

pervious code with integration conflicts might try to sort 



import gradio as gr
from PIL import Image
from super_image import HanModel, ImageLoader


# ---------------- Model setup ----------------

# Supported upscale factors for HAN
AVAILABLE_SCALES = ["2x", "3x", "4x"]


def load_sr_models():
    """
    Load HAN super‑resolution models once at startup.

    Returns
    -------
    dict[str, HanModel]
        Mapping from scale label (e.g. "4x") to initialized model.
    """
    models = {}
    for label in AVAILABLE_SCALES:
        scale = int(label[0])  # "4x" -> 4, "2x" -> 2, etc.
        models[label] = HanModel.from_pretrained("eugenesiow/han", scale=scale)
    return models


SR_MODELS = load_sr_models()


# ---------------- Inference logic ----------------

def run_han_sr(input_image: Image.Image, scale_factor: str):
    """
    Apply HAN super‑resolution to a PIL image.

    Parameters
    ----------
    input_image : PIL.Image.Image
        Input low‑resolution image from the Gradio UI.
    scale_factor : str
        One of "2x", "3x", "4x" indicating the desired upscale.

    Returns
    -------
    combined : PIL.Image.Image
        Side‑by‑side image with original on the left and SR result on the right.
    info_text : str
        Human‑readable summary of input and output resolutions.
    """
    if input_image is None:
        return None, "upload an image first."

    model = SR_MODELS[scale_factor]

    # Convert PIL -> tensor expected by super_image
    inputs = ImageLoader.load_image(input_image)

    # Run the model
    preds = model(inputs)

    # Convert tensor -> PIL and save to memory
    sr_path = "temp_sr.png"
    ImageLoader.save_image(preds, sr_path)
    sr_image = Image.open(sr_path).convert("RGB")

    # Build side‑by‑side comparison image
    w_in, h_in = input_image.size
    w_sr, h_sr = sr_image.size
    combined = Image.new("RGB", (w_in + w_sr, max(h_in, h_sr)))
    combined.paste(input_image, (0, 0))
    combined.paste(sr_image, (w_in, 0))

    info_text = (
        f"Input: {w_in} x {h_in}  |  "
        f"Super‑resolved: {w_sr} x {h_sr}  ({scale_factor} upscaling)"
    )

    return combined, info_text


# ---------------- Gradio interface ----------------

with gr.Blocks(title="Image Super‑Resolution (HAN)") as demo:
    gr.Markdown(
        "## Image Super‑Resolution (HAN)\n"
        "Upload a low‑resolution image, select an upscale factor, and compare "
        "the original and HAN super‑resolved results side by side."
    )

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload low‑res image")
            scale_radio = gr.Radio(
                choices=AVAILABLE_SCALES,
                value="4x",
                label="Upscale factor",
            )
            run_button = gr.Button("Run Super‑Resolution")

        with gr.Column():
            output_img = gr.Image(
                type="pil",
                label="Original (left) / Super‑resolved (right)",
            )
            info_box = gr.Textbox(
                label="Image info",
                interactive=False,
            )

    run_button.click(
        fn=run_han_sr,
        inputs=[input_img, scale_radio],
        outputs=[output_img, info_box],
    )


if __name__ == "__main__":
    demo.launch()
'''