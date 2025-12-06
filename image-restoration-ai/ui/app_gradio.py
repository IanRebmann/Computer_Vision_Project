import gradio as gr
from PIL import Image, ImageOps
import time

from src.config import load_config
from src.enhancer import ImageEnhancer

def extract_mask_from_editor(editor_value):
    """
    Gradio ImageEditor returns a dict-like value with:
      - background
      - layers
      - composite

    We treat the first layer (if present) as the user's mask strokes.
    """
    if editor_value is None:
        return None

    # Depending on version, editor_value may be a dict
    # with keys: "background", "layers", "composite"
    layers = None
    if isinstance(editor_value, dict):
        layers = editor_value.get("layers", None)

    if not layers:
        # Fallback: if no layers, try composite
        comp = editor_value.get("composite") if isinstance(editor_value, dict) else None
        if comp is None:
            return None
        # Convert composite to grayscale mask
        return comp.convert("L")

    # Use the first drawn layer as mask
    layer0 = layers[0]

    # Some versions may give RGBA; convert to L
    if layer0.mode != "L":
        layer0 = layer0.convert("L")

    # Binarize so mask is clean
    layer0 = layer0.point(lambda p: 255 if p > 10 else 0)

    return layer0


cfg = load_config("configs/default.yaml")
enhancer = ImageEnhancer(cfg)
def run_task(task, image, mask, prompt, negative_prompt, strength, guidance, steps, seed, use_lora, lora_scale, progress=gr.Progress()):
    if image is None:
        return None

    progress(0.05, desc="Preparing inputs")

    base_kwargs = {
        "prompt": prompt or "",
        "negative_prompt": negative_prompt or "",
        "guidance_scale": guidance,
        "num_inference_steps": int(steps),
        "seed": int(seed) if seed and seed >= 0 else None,
    }

    if task == "denoise":
        progress(0.2, desc="Running denoise model")
        return enhancer.run_denoise(image, **{**base_kwargs, "strength": strength})

    if task == "superres":
        progress(0.2, desc="Running super-resolution model")
        return enhancer.run_superres(image)

    if task == "colorize":
        progress(0.2, desc="Running colorization model")
        return enhancer.run_colorize(image, **{**base_kwargs, "strength": strength})

    if task == "inpaint":
        progress(0.15, desc="Extracting mask")
        mask_img = extract_mask_from_editor(mask) if isinstance(mask, dict) else mask
        if mask_img is None:
            raise gr.Error("Mask required for inpainting.")

        progress(0.25, desc="Running inpainting model")
        inpaint_kwargs = {
            **base_kwargs,
            "use_lora": bool(use_lora),
            "lora_scale": float(lora_scale),
        }
        return enhancer.run_inpaint(image, mask_img, **inpaint_kwargs)

    return image

with gr.Blocks() as demo:
    gr.Markdown("# Image Restoration & Enhancement (Hugging Face)")

    with gr.Row():
        image_in = gr.Image(type="pil", label="Input Image")
        mask_in  = gr.ImageEditor(type="pil", label="Draw Mask (use brush)")

    task = gr.Radio(["denoise", "superres", "colorize", "inpaint"], value="denoise", label="Task")

    with gr.Accordion("Advanced Settings", open=False):
        prompt = gr.Textbox(value=cfg.defaults.get("denoise", {}).get("prompt", ""), label="Prompt")
        negative_prompt = gr.Textbox(value="", label="Negative Prompt")

        strength = gr.Slider(
            0.05, 0.9,
            value=0.3,
            step=0.01,
            label="Strength (denoise/colorize only)"
        )

        guidance = gr.Slider(1.0, 12.0, value=6.0, step=0.1, label="Guidance Scale")
        steps = gr.Slider(10, 50, value=20, step=1, label="Inference Steps")
        seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")

        use_lora = gr.Checkbox(value=True, label="Use Inpainting LoRA (if available)")
        lora_scale = gr.Slider(
            0.0, 1.5,
            value=cfg.defaults.get("inpaint", {}).get("lora_scale", 0.7),
            step=0.05,
            label="Inpainting LoRA Strength"
        )

    run_btn = gr.Button("Run")
    image_out = gr.Image(type="pil", label="Output")

    run_btn.click(
        fn=run_task,
        inputs=[task, image_in, mask_in, prompt, negative_prompt, strength, guidance, steps, seed, use_lora, lora_scale],
        outputs=image_out
    )

if __name__ == "__main__":
    demo.launch()
