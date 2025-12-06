import os
import sys
from pathlib import Path

# Add project root to PYTHONPATH so `import src...` works
ROOT = Path(__file__).resolve().parents[1]  # ui/ -> project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import dataclass
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import torch

# Your project imports
from src.config import load_config
from src.enhancer import ImageEnhancer

# Canvas component (Gradio ImageEditor equivalent)
from streamlit_drawable_canvas import st_canvas


# -----------------------------
# Mask helpers (same spirit as Gradio version)
# -----------------------------

def make_faded_background(img: Image.Image, fade=0.55):
    """
    Create a lighter/faded background so white strokes stand out.
    fade=0.55 means 55% original + 45% white.
    """
    if img is None:
        return None
    img = img.convert("RGB")
    white = Image.new("RGB", img.size, "white")
    return Image.blend(img, white, alpha=(1.0 - fade))


def normalize_mask(mask: Image.Image, target_size):
    """
    Ensure mask is clean binary L-mode matching target size.
    White = fill, Black = keep.
    """
    if mask is None:
        return None

    mask = mask.convert("L")
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.NEAREST)

    mask = mask.point(lambda p: 255 if p > 127 else 0)
    return mask


def mask_from_canvas_rgba(rgba: np.ndarray):
    """
    Convert canvas RGBA array -> L-mode mask (white where drawn).
    We use alpha channel to detect strokes.
    """
    if rgba is None:
        return None

    alpha = rgba[..., 3]
    m = (alpha > 0).astype(np.uint8) * 255
    return Image.fromarray(m, mode="L")


def get_active_mask(
    image: Image.Image,
    mask_source: str,
    mask_upload: Optional[Image.Image],
    drawn_mask: Optional[Image.Image],
    invert: bool = False,
):
    if image is None:
        return None

    if mask_source == "upload" and mask_upload is not None:
        m = normalize_mask(mask_upload, image.size)
    else:
        m = normalize_mask(drawn_mask, image.size) if drawn_mask is not None else None

    if m is None:
        return None

    if invert:
        m = ImageOps.invert(m)

    return m


# -----------------------------
# Load config + enhancer
# -----------------------------

st.set_page_config(page_title="Image Restoration & Enhancement", layout="wide")


@st.cache_resource
def get_cfg():
    return load_config("configs/default.yaml")

@st.cache_resource
def get_enhancer():
    cfg = get_cfg()
    return ImageEnhancer(cfg)

cfg = get_cfg()
enhancer = get_enhancer()

# -----------------------------
# Optional warmup (safe + fast)
# -----------------------------

def maybe_warmup():
    if st.session_state.get("_did_warmup", False):
        return
    try:
        st.write("Warmup: running quick inpaint...")
        img = Image.new("RGB", (512, 512), "gray")
        mask = Image.new("L", (512, 512), 0)
        enhancer.run_inpaint(
            img, mask,
            prompt="warmup",
            num_inference_steps=3,
            guidance_scale=3.0,
            use_lora=False
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        st.session_state["_did_warmup"] = True
        st.write("Warmup done.")
    except Exception as e:
        st.session_state["_did_warmup"] = True
        st.write(f"Warmup skipped: {e}")

# Comment out if you don't want this shown in UI
maybe_warmup()


# -----------------------------
# UI
# -----------------------------

st.title("Image Restoration & Enhancement (Hugging Face)")
st.caption("**Mask rule:** White = area to fill/repair, Black = keep unchanged.")

# Task (keeping your structure)
task = st.radio(
    "Task",
    ["denoise", "superres", "colorize", "inpaint"],
    index=3,  # default to inpaint
    horizontal=True
)

# Inputs
col_img, col_mask = st.columns([1, 1])

with col_img:
    img_file = st.file_uploader("Input Image", type=["png", "jpg", "jpeg"])

with col_mask:
    mask_source = st.radio("Mask Source", ["draw", "upload"], horizontal=True)
    invert_mask = st.checkbox("Invert Mask", value=False)

    mask_file = None
    if mask_source == "upload":
        mask_file = st.file_uploader("Upload Mask (optional)", type=["png", "jpg", "jpeg"])

# Load PIL images
image = Image.open(img_file).convert("RGB") if img_file else None
mask_upload = Image.open(mask_file) if mask_file else None


# -----------------------------
# Draw mask section
# -----------------------------

drawn_mask = None

if image is not None and mask_source == "draw":
    st.subheader("Draw Mask")

    # Display scaling (keeps canvas responsive)
    w, h = image.size
    display_max = 900
    scale = min(display_max / max(w, h), 1.0)
    disp_w = max(64, int(w * scale))
    disp_h = max(64, int(h * scale))

    faded = make_faded_background(image).resize((disp_w, disp_h), Image.LANCZOS)

    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,1.0)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_image=faded,
        update_streamlit=True,
        height=disp_h,
        width=disp_w,
        drawing_mode="freedraw",
        key="mask_canvas",
    )

    if canvas_result.image_data is not None:
        m_small = mask_from_canvas_rgba(canvas_result.image_data.astype(np.uint8))
        if m_small is not None:
            # Resize back to original
            drawn_mask = m_small.resize(image.size, Image.NEAREST)


# -----------------------------
# Mask preview
# -----------------------------

st.subheader("Mask Preview (final binary)")

preview_btn = st.button("Update Mask Preview")

# Keep last preview in session state
if "mask_preview" not in st.session_state:
    st.session_state["mask_preview"] = None

if preview_btn and image is not None:
    m = get_active_mask(
        image=image,
        mask_source=mask_source,
        mask_upload=mask_upload,
        drawn_mask=drawn_mask,
        invert=invert_mask
    )
    if m is None:
        m = Image.new("L", image.size, 0)
    st.session_state["mask_preview"] = m

# Show preview if available
if image is not None:
    if st.session_state["mask_preview"] is None:
        st.image(Image.new("L", image.size, 0), caption="(no preview yet)", use_column_width=True)
    else:
        st.image(st.session_state["mask_preview"], use_column_width=True)


# -----------------------------
# Advanced settings
# -----------------------------

with st.expander("Advanced Settings", expanded=False):
    prompt = st.text_input(
        "Prompt",
        value=cfg.defaults.get("denoise", {}).get("prompt", "")
    )
    negative_prompt = st.text_input("Negative Prompt", value="")

    strength = st.slider(
        "Strength (denoise/colorize only)",
        0.05, 0.9, 0.3, 0.01
    )

    guidance = st.slider("Guidance Scale", 1.0, 12.0, 6.0, 0.1)
    steps = st.slider("Inference Steps", 10, 50, 20, 1)
    seed = st.number_input("Seed (-1 random)", value=-1, step=1)

    use_lora = st.checkbox("Use Inpainting LoRA (if available)", value=True)
    lora_scale = st.slider(
        "Inpainting LoRA Strength",
        0.0, 1.5,
        float(cfg.defaults.get("inpaint", {}).get("lora_scale", 0.7)),
        0.05
    )

# Defaults if expander not opened yet
if "prompt" not in locals():
    prompt = ""
    negative_prompt = ""
    strength = 0.3
    guidance = 6.0
    steps = 20
    seed = -1
    use_lora = True
    lora_scale = 0.7


# -----------------------------
# Run
# -----------------------------

run_btn = st.button("Run")

if run_btn:
    if image is None:
        st.error("Please upload an input image.")
        st.stop()

    base_kwargs = {
        "prompt": prompt or "",
        "negative_prompt": negative_prompt or "",
        "guidance_scale": float(guidance),
        "num_inference_steps": int(steps),
        "seed": int(seed) if seed is not None and seed >= 0 else None,
    }

    # You commented these in Gradio; keeping same behavior:
    # if task == "denoise":
    #     denoise_kwargs = {**base_kwargs, "strength": float(strength)}
    #     out = enhancer.run_denoise(image, **denoise_kwargs)
    # elif task == "superres":
    #     out = enhancer.run_superres(image)
    # elif task == "colorize":
    #     color_kwargs = {**base_kwargs, "strength": float(strength)}
    #     out = enhancer.run_colorize(image, **color_kwargs)
    # else ...

    if task == "inpaint":
        st.write("Resolving mask...")

        m = get_active_mask(
            image=image,
            mask_source=mask_source,
            mask_upload=mask_upload,
            drawn_mask=drawn_mask,
            invert=invert_mask
        )
        if m is None:
            st.error("Mask required for inpainting.")
            st.stop()

        inpaint_kwargs = {
            **base_kwargs,
            "use_lora": bool(use_lora),
            "lora_scale": float(lora_scale),
        }

        with st.spinner("Running inpaint..."):
            out = enhancer.run_inpaint(image, m, **inpaint_kwargs)

        st.subheader("Output")
        st.image(out, use_column_width=True)

    else:
        st.info("Only inpaint is wired here (matching your current Gradio file).")
        st.image(image, caption="Input", use_column_width=True)


# -----------------------------
# Input previews
# -----------------------------

st.write("---")
st.subheader("Input Preview")

c1, c2 = st.columns(2)
with c1:
    if image is not None:
        st.image(image, caption=f"Image: {image.size}", use_column_width=True)
    else:
        st.write("No image loaded.")
with c2:
    if mask_upload is not None:
        st.image(mask_upload, caption="Uploaded mask", use_column_width=True)
    elif drawn_mask is not None:
        st.image(drawn_mask, caption="Drawn mask", use_column_width=True)
    else:
        st.write("No mask loaded/drawn yet.")
