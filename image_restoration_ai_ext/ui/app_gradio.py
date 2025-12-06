import gradio as gr
from PIL import Image, ImageOps
import os
from dotenv import load_dotenv

# Load environment variables from the .env file (if present)
load_dotenv()

from image_restoration_ai_ext.src.config import load_config
from image_restoration_ai_ext.src.enhancer import ImageEnhancer

# -----------------------------
# Mask helpers
# -----------------------------

def make_faded_background(img: Image.Image, fade=0.55):
    """
    Create a lighter/faded background for the editor so strokes stand out.
    fade=0.55 means 55% original + 45% white.
    """
    if img is None:
        return None
    img = img.convert("RGB")
    white = Image.new("RGB", img.size, "white")
    # alpha here is how much of 'white' to blend in
    return Image.blend(img, white, alpha=(1.0 - fade))


def extract_mask_from_editor(editor_value):
    """
    Gradio ImageEditor returns a dict-like value with:
      - background
      - layers
      - composite

    We treat the first layer (if present) as the user's mask strokes.
    Falls back to composite when layers aren't available.
    Returns a PIL L-mode mask.
    """
    if editor_value is None:
        return None

    layers = None
    if isinstance(editor_value, dict):
        layers = editor_value.get("layers", None)

    if layers and len(layers) > 0 and layers[0] is not None:
        layer0 = layers[0]
        if layer0.mode != "L":
            layer0 = layer0.convert("L")
        # Light binarization
        layer0 = layer0.point(lambda p: 255 if p > 10 else 0)
        return layer0

    # Fallback: composite
    if isinstance(editor_value, dict):
        comp = editor_value.get("composite", None)
        if comp is not None:
            comp = comp.convert("L")
            comp = comp.point(lambda p: 255 if p > 10 else 0)
            return comp

    return None


def normalize_mask(mask: Image.Image, target_size):
    """
    Ensure mask is clean binary L-mode matching target size.
    """
    if mask is None:
        return None

    mask = mask.convert("L")
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.NEAREST)

    # Strict black/white
    mask = mask.point(lambda p: 255 if p > 127 else 0)
    return mask


def init_editor_value(image: Image.Image):
    """
    When an image is loaded, set the editor background to a faded version.
    Start with no layers.
    """
    if image is None:
        return None

    bg = make_faded_background(image)
    return {"background": bg, "layers": [], "composite": bg}


def editor_with_uploaded_mask(image: Image.Image, mask: Image.Image):
    """
    Try to place uploaded mask as a visible layer in the editor.
    If a Gradio version doesn't support this well, preview still works regardless.
    """
    if image is None:
        return None

    bg = make_faded_background(image)
    layers = []

    if mask is not None:
        m = normalize_mask(mask, image.size)
        # Make a visible overlay layer
        m_rgb = ImageOps.colorize(m, black="black", white="white").convert("RGBA")
        layers = [m_rgb]

    return {"background": bg, "layers": layers, "composite": bg}


def get_active_mask(image, mask_source, mask_upload, mask_editor, invert=False):
    """
    Decide which mask to use based on selection.
    """
    if image is None:
        return None

    if mask_source == "upload" and mask_upload is not None:
        m = normalize_mask(mask_upload, image.size)
    else:
        drawn = extract_mask_from_editor(mask_editor)
        m = normalize_mask(drawn, image.size) if drawn is not None else None

    if m is None:
        return None

    if invert:
        m = ImageOps.invert(m)

    return m


def update_mask_preview(image, mask_source, mask_upload, mask_editor, invert):
    """
    Returns a solid black/white preview of the final computed mask.
    """
    if image is None:
        return None

    m = get_active_mask(image, mask_source, mask_upload, mask_editor, invert=invert)
    if m is None:
        return Image.new("L", image.size, 0)
    return m


# Optional speed guard (uncomment if needed)
# def resize_for_sd(img: Image.Image, mask: Image.Image, max_side=768):
#     w, h = img.size
#     scale = min(max_side / max(w, h), 1.0)
#     nw, nh = int(w * scale), int(h * scale)
#     nw = max(64, (nw // 8) * 8)
#     nh = max(64, (nh // 8) * 8)
#     if (nw, nh) != (w, h):
#         img = img.resize((nw, nh), Image.LANCZOS)
#         mask = mask.resize((nw, nh), Image.NEAREST)
#     return img, mask


# -----------------------------
# Load config + enhancer
# -----------------------------
cfg = load_config(f"{os.getenv('ROOT_PATH')}/configs/default.yaml")
enhancer = ImageEnhancer(cfg)

# -----------------------------
# Task runner
# -----------------------------

def run_task(
    task,
    image,
    mask_editor,
    mask_source,
    mask_upload,
    invert_mask,
    prompt,
    negative_prompt,
    strength,
    guidance,
    steps,
    seed,
    use_lora,
    lora_scale,
    progress=gr.Progress()
):
    print(f"[UI] Running task={task}", flush=True)
    if image is None:
        return None

    print("[UI] image pixels:", image.size, "mode:", image.mode)

    progress(0.05, desc="Preparing inputs")
    base_kwargs = {
        "prompt": prompt or "",
        "negative_prompt": negative_prompt or "",
        "guidance_scale": float(guidance),
        "num_inference_steps": int(steps),
        "seed": int(seed) if seed is not None and seed >= 0 else None,
    }

    '''if task == "denoise":
        progress(0.2, desc="Running denoise")
        denoise_kwargs = {**base_kwargs, "strength": float(strength)}
        return enhancer.run_denoise(image, **denoise_kwargs)

    if task == "superres":
        progress(0.2, desc="Running super-resolution")
        return enhancer.run_superres(image)

    if task == "colorize":
        progress(0.2, desc="Running colorize")
        color_kwargs = {**base_kwargs, "strength": float(strength)}
        return enhancer.run_colorize(image, **color_kwargs)'''

    if task == "inpaint":
        progress(0.15, desc="Resolving mask")

        print("[UI] mask_source:", mask_source)
        mask_img = get_active_mask(
            image=image,
            mask_source=mask_source,
            mask_upload=mask_upload,
            mask_editor=mask_editor,
            invert=bool(invert_mask)
        )
        
        if mask_img is None:
            raise gr.Error("Mask required for inpainting.")

        # Optional speed safety:
        # image, mask_img = resize_for_sd(image, mask_img, max_side=768)

        progress(0.25, desc="Running inpaint")

        inpaint_kwargs = {
            **base_kwargs,
            "use_lora": bool(use_lora),
            "lora_scale": float(lora_scale),
        }
        return enhancer.run_inpaint(image, mask_img, **inpaint_kwargs)

    return image


# -----------------------------
# UI
# -----------------------------

with gr.Blocks() as demo:
    gr.Markdown("# Image Restoration & Enhancement (Hugging Face)")
    gr.Markdown("**Mask rule:** White = area to fill/repair, Black = keep unchanged.")

    # Input image
    with gr.Row():
        image_in = gr.Image(type="pil", label="Input Image")

    # Mask source selection
    with gr.Row():
        mask_source = gr.Radio(
            ["draw", "upload"],
            value="draw",
            label="Mask Source"
        )
        invert_mask = gr.Checkbox(value=False, label="Invert Mask")

    # Mask inputs
    with gr.Row():
        mask_upload = gr.Image(type="pil", label="Upload Mask (optional)", visible=False)
        mask_in = gr.ImageEditor(type="pil", label="Draw Mask (white = fill)", visible=True)

    # Mask preview
    with gr.Row():
        mask_preview = gr.Image(type="pil", label="Mask Preview (final binary)")

    # Task selection
    task = gr.Radio(
        ["denoise", "superres", "colorize", "inpaint"],
        value="inpaint",
        label="Task"
    )

    # Advanced settings
    with gr.Accordion("Advanced Settings", open=False):
        prompt = gr.Textbox(
            value=cfg.defaults.get("denoise", {}).get("prompt", ""),
            label="Prompt"
        )
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

    # Run + output
    run_btn = gr.Button("Run")
    image_out = gr.Image(type="pil", label="Output")

    # -----------------------------
    # UI event wiring
    # -----------------------------

    def toggle_mask_inputs(source):
        return (
            gr.update(visible=(source == "upload")),
            gr.update(visible=(source == "draw")),
        )

    mask_source.change(
        fn=toggle_mask_inputs,
        inputs=[mask_source],
        outputs=[mask_upload, mask_in]
    )

    # When image changes, initialize editor background and preview
    def on_image_change(img, source, uploaded, invert):
        if img is None:
            return None, None

        editor_val = init_editor_value(img)

        if source == "upload" and uploaded is not None:
            editor_val = editor_with_uploaded_mask(img, uploaded)

        preview = update_mask_preview(img, source, uploaded, editor_val, invert)
        return editor_val, preview

    image_in.change(
        fn=on_image_change,
        inputs=[image_in, mask_source, mask_upload, invert_mask],
        outputs=[mask_in, mask_preview]
    )

    # When uploaded mask changes, refresh preview and optionally overlay
    def on_mask_upload_change(img, source, uploaded, editor_val, invert):
        if img is None:
            return editor_val, None

        if source == "upload":
            editor_val = editor_with_uploaded_mask(img, uploaded)

        preview = update_mask_preview(img, source, uploaded, editor_val, invert)
        return editor_val, preview

    mask_upload.change(
        fn=on_mask_upload_change,
        inputs=[image_in, mask_source, mask_upload, mask_in, invert_mask],
        outputs=[mask_in, mask_preview]
    )

    # When editor changes, update preview
    mask_in.change(
        fn=update_mask_preview,
        inputs=[image_in, mask_source, mask_upload, mask_in, invert_mask],
        outputs=[mask_preview]
    )

    invert_mask.change(
        fn=update_mask_preview,
        inputs=[image_in, mask_source, mask_upload, mask_in, invert_mask],
        outputs=[mask_preview]
    )

    # -----------------------------
    # Run button binding
    # -----------------------------

    run_btn.click(
        fn=run_task,
        inputs=[
            task,
            image_in,
            mask_in,
            mask_source,
            mask_upload,
            invert_mask,
            prompt,
            negative_prompt,
            strength,
            guidance,
            steps,
            seed,
            use_lora,
            lora_scale,
        ],
        outputs=image_out
    )


if __name__ == "__main__":
    demo.launch()
