import sys
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import torch

# Add project root to PYTHONPATH so `import src...` works
ROOT = Path(__file__).resolve().parents[1]  # ui/ -> project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.enhancer import ImageEnhancer

from streamlit_drawable_canvas import st_canvas


# -----------------------------
# UI sizing constants
# -----------------------------
PREVIEW_W = 320     # consistent small previews
OUTPUT_W = 900      # main output size
FULL_W = 480        # full input objects display size


# -----------------------------
# Mask helpers
# -----------------------------

def make_faded_background(img: Image.Image, fade=0.55):
    """
    Create a lighter/faded background so strokes stand out.
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

    # strict black/white
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


def resize_for_preview(img: Image.Image, width=PREVIEW_W):
    """
    Standard-size preview while preserving aspect ratio.
    """
    if img is None:
        return None
    w, h = img.size
    if w <= width:
        return img
    scale = width / float(w)
    nh = int(h * scale)
    return img.resize((width, nh), Image.LANCZOS)


def rgba_from_hex(hex_color: str, opacity: float):
    """
    Convert #RRGGBB + opacity -> rgba(r,g,b,a)
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


# -----------------------------
# Streamlit setup
# -----------------------------

st.set_page_config(
    page_title="Image Restoration & Enhancement",
    layout="wide"
)

# -----------------------------
# Load config + enhancer
# -----------------------------

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
# Optional warmups
# -----------------------------

def maybe_warmup_inpaint():
    if st.session_state.get("_did_inpaint_warmup", False):
        return
    try:
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
    except Exception:
        pass
    st.session_state["_did_inpaint_warmup"] = True


def maybe_warmup_superres():
    if st.session_state.get("_did_superres_warmup", False):
        return
    try:
        img = Image.new("RGB", (128, 128), "gray")
        enhancer.run_superres(img, mode="Crystal clear (pretrained)")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    st.session_state["_did_superres_warmup"] = True


# You can comment these out if you don't want warmups
maybe_warmup_inpaint()
maybe_warmup_superres()


# -----------------------------
# Header
# -----------------------------

st.title("Image Restoration & Enhancement (Hugging Face)")
st.caption("**Mask rule:** White = area to fill/repair, Black = keep unchanged.")


# -----------------------------
# Task selection
# -----------------------------

task = st.radio(
    "Task",
    ["superres", "inpaint", "denoise", "colorize"],
    index=0,
    horizontal=True
)

# -----------------------------
# Inputs row
# -----------------------------

col_img, col_mask = st.columns([1, 1])

with col_img:
    img_file = st.file_uploader("Input Image", type=["png", "jpg", "jpeg"])

with col_mask:
    # Only relevant primarily for inpaint
    mask_source = st.radio("Mask Source", ["draw", "upload"], horizontal=True)
    invert_mask = st.checkbox("Invert Mask", value=False)

    mask_file = None
    if mask_source == "upload":
        mask_file = st.file_uploader("Upload Mask (optional)", type=["png", "jpg", "jpeg"])


# Load PIL images
image = Image.open(img_file).convert("RGB") if img_file else None
mask_upload = Image.open(mask_file) if mask_file else None


# -----------------------------
# Mode selector for superres
# -----------------------------

superres_mode = cfg.defaults.get("superres", {}).get("mode", "Crystal clear (pretrained)")

if task == "superres":
    superres_mode = st.selectbox(
        "Super-Resolution Style",
        ["Crystal clear (pretrained)", "Smooth (fine-tuned)"],
        index=0 if superres_mode != "Smooth (fine-tuned)" else 1
    )


# -----------------------------
# Standard-sized previews
# -----------------------------

st.subheader("Input & Mask Previews (standard-sized)")

p1, p2 = st.columns(2)

with p1:
    if image is not None:
        st.image(
            resize_for_preview(image),
            caption=f"Input preview (orig: {image.size})",
            width=PREVIEW_W
        )
    else:
        st.write("No image loaded.")

with p2:
    if mask_upload is not None:
        st.image(
            resize_for_preview(mask_upload),
            caption="Uploaded mask preview",
            width=PREVIEW_W
        )
    else:
        st.write("No uploaded mask.")


# -----------------------------
# Brush controls (for draw mode)
# -----------------------------

st.subheader("Brush Settings")

b1, b2, b3 = st.columns([1, 1, 1])

with b1:
    brush_size = st.slider("Brush size", 1, 80, 20, 1)

with b2:
    brush_color = st.color_picker("Brush color", "#FFFFFF")

with b3:
    brush_opacity = st.slider("Brush opacity", 0.1, 1.0, 1.0, 0.05)

stroke_rgba = rgba_from_hex(brush_color, brush_opacity)


# -----------------------------
# Draw mask section
# Canvas on its own row, sized to the image
# -----------------------------

drawn_mask = None

if image is not None and mask_source == "draw":
    st.subheader("Draw Mask (full image size)")

    w, h = image.size
    faded = make_faded_background(image)

    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",   # transparent fill
        stroke_width=int(brush_size),
        stroke_color=stroke_rgba,
        background_image=faded,             # assuming your community fix works
        update_streamlit=True,
        height=h,
        width=w,
        drawing_mode="freedraw",
        key="mask_canvas",
    )

    if canvas_result.image_data is not None:
        m_full = mask_from_canvas_rgba(canvas_result.image_data.astype(np.uint8))
        if m_full is not None:
            drawn_mask = normalize_mask(m_full, image.size)


# -----------------------------
# Mask preview (final binary)
# -----------------------------

st.subheader("Mask Preview (final binary)")

preview_btn = st.button("Update Mask Preview")

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

if image is not None:
    show_mask = st.session_state["mask_preview"]
    if show_mask is None:
        show_mask = Image.new("L", image.size, 0)

    st.image(
        resize_for_preview(show_mask),
        caption="Final mask preview (standard-sized)",
        width=PREVIEW_W
    )


# -----------------------------
# Advanced settings
# -----------------------------

with st.expander("Advanced Settings", expanded=False):
    # Prompts apply to SD tasks (inpaint/denoise/colorize)
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

# Safeguards if expander not opened
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

    # --- SUPER-RESOLUTION ---
    if task == "superres":
        with st.spinner("Running Swin2SR x4..."):
            out = enhancer.run_superres(image, mode=superres_mode)

        st.subheader("Output (Super-Resolution x4)")
        st.image(out, width=OUTPUT_W)

    # --- INPAINT ---
    elif task == "inpaint":
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

        st.subheader("Output (Inpainting)")
        st.image(out, width=OUTPUT_W)

    # --- Others (placeholders if you want to wire later) ---
    elif task == "denoise":
        st.info("Denoise is not wired in this UI block yet.")
        st.image(image, caption="Input", width=FULL_W)

    elif task == "colorize":
        st.info("Colorize is not wired in this UI block yet.")
        st.image(image, caption="Input", width=FULL_W)


# -----------------------------
# Full input objects (original sizes)
# -----------------------------

st.write("---")
st.subheader("Full Input Objects (original sizes)")

c1, c2 = st.columns(2)

with c1:
    if image is not None:
        st.image(image, caption=f"Image: {image.size}", width=FULL_W)
    else:
        st.write("No image loaded.")

with c2:
    if mask_upload is not None:
        st.image(mask_upload, caption=f"Uploaded mask: {mask_upload.size}", width=FULL_W)
    elif drawn_mask is not None:
        st.image(drawn_mask, caption=f"Drawn mask: {drawn_mask.size}", width=FULL_W)
    else:
        st.write("No mask loaded/drawn yet.")


# -----------------------------
# Debug / utilities
# -----------------------------

with st.expander("Debug / Utilities", expanded=False):
    st.write("Device:", cfg.device)
    st.write("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        st.write("GPU:", torch.cuda.get_device_name(0))

    if st.button("Clear GPU Cache"):
        try:
            torch.cuda.empty_cache()
            st.success("Cleared CUDA cache.")
        except Exception as e:
            st.warning(f"Could not clear cache: {e}")
