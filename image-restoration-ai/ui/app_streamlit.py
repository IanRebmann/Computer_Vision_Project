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
PREVIEW_W = 300     # consistent small previews
OUTPUT_W = 920      # main output size
FULL_W = 460        # full input objects display size


# -----------------------------
# Mask helpers
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
    layout="wide",
)

# Subtle app styling
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.3rem; padding-bottom: 2rem; }
      h1, h2, h3 { letter-spacing: -0.3px; }
      .soft-card {
          background: rgba(255,255,255,0.03);
          border: 1px solid rgba(255,255,255,0.06);
          border-radius: 14px;
          padding: 1.0rem 1.1rem;
          margin-bottom: 1rem;
      }
      .tiny-muted {
          opacity: 0.7;
          font-size: 0.9rem;
      }
      .section-title {
          font-size: 1.05rem;
          font-weight: 600;
          margin-bottom: 0.35rem;
      }
      .hr {
          height: 1px;
          background: rgba(255,255,255,0.08);
          margin: 1.0rem 0 1.2rem 0;
      }
    </style>
    """,
    unsafe_allow_html=True
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
        try:
            enhancer.run_superres(img, mode="Crystal clear (pretrained)")
        except TypeError:
            enhancer.run_superres(img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    st.session_state["_did_superres_warmup"] = True


maybe_warmup_inpaint()
maybe_warmup_superres()


# -----------------------------
# Header
# -----------------------------

st.markdown("## 🧩 Image Restoration & Enhancement")
st.markdown(
    "<div class='tiny-muted'>"
    "Unified UI for <b>Super-Resolution</b>, <b>Inpainting</b>, "
    "<b>Denoising</b>, and <b>Colorization</b>. "
    "Mask rule: <b>White</b> = repair region, <b>Black</b> = preserve."
    "</div>",
    unsafe_allow_html=True
)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# -----------------------------
# Sidebar controls
# -----------------------------

with st.sidebar:
    st.markdown("### Controls")

    task = st.radio(
        "Task",
        ["superres", "inpaint", "denoise", "colorize"],
        index=0
    )

    st.markdown("#### Input")
    img_file = st.file_uploader("Input image", type=["png", "jpg", "jpeg"])

    image = Image.open(img_file).convert("RGB") if img_file else None

    # defaults so app never crashes if task changes
    mask_source = "draw"
    invert_mask = False
    mask_file = None
    mask_upload = None

    if task == "inpaint":
        st.markdown("#### Mask")
        mask_source = st.radio("Mask source", ["draw", "upload"])
        invert_mask = st.checkbox("Invert mask", value=False)

        if mask_source == "upload":
            mask_file = st.file_uploader("Upload mask", type=["png", "jpg", "jpeg"])
            mask_upload = Image.open(mask_file) if mask_file else None

        st.markdown("#### Brush")
        brush_size = st.slider("Brush size", 1, 80, 20, 1)
        brush_color = st.color_picker("Brush color", "#FFFFFF")
        brush_opacity = st.slider("Brush opacity", 0.1, 1.0, 1.0, 0.05)
    else:
        # still define values to avoid NameError
        brush_size = 20
        brush_color = "#FFFFFF"
        brush_opacity = 1.0

    # Superres mode (only relevant when selected)
    superres_mode = cfg.defaults.get("superres", {}).get("mode", "Crystal clear (pretrained)")
    if task == "superres":
        st.markdown("#### Super-Resolution Style")
        superres_mode = st.selectbox(
            "Style",
            ["Crystal clear (pretrained)", "Smooth (fine-tuned)"],
            index=0 if superres_mode != "Smooth (fine-tuned)" else 1
        )

    with st.expander("Advanced Settings", expanded=False):
        prompt = st.text_input(
            "Prompt",
            value=cfg.defaults.get("denoise", {}).get("prompt", "")
        )
        negative_prompt = st.text_input("Negative Prompt", value="")

        strength = st.slider(
            "Strength (denoise/colorize)",
            0.05, 0.9, 0.3, 0.01
        )

        guidance = st.slider("Guidance Scale", 1.0, 12.0, 6.0, 0.1)
        steps = st.slider("Inference Steps", 10, 50, 20, 1)
        seed = st.number_input("Seed (-1 random)", value=-1, step=1)

        use_lora = st.checkbox("Use Inpainting LoRA", value=True)
        lora_scale = st.slider(
            "Inpainting LoRA Strength",
            0.0, 1.5,
            float(cfg.defaults.get("inpaint", {}).get("lora_scale", 0.7)),
            0.05
        )

    # Safe fallbacks if expander never opened
    if "prompt" not in locals():
        prompt = ""
        negative_prompt = ""
        strength = 0.3
        guidance = 6.0
        steps = 20
        seed = -1
        use_lora = True
        lora_scale = 0.7

    st.markdown("---")
    run_btn = st.button("🚀 Run", use_container_width=True)

    with st.expander("Debug / Utilities", expanded=False):
        st.write("Device:", getattr(cfg, "device", "unknown"))
        st.write("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            st.write("GPU:", torch.cuda.get_device_name(0))
        if st.button("Clear GPU Cache"):
            try:
                torch.cuda.empty_cache()
                st.success("Cleared CUDA cache.")
            except Exception as e:
                st.warning(f"Could not clear cache: {e}")


# -----------------------------
# Main area: previews + canvas + output
# -----------------------------

# Standard input previews card
with st.container():
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Input Previews</div>", unsafe_allow_html=True)

    p1, p2 = st.columns(2)

    with p1:
        if image is not None:
            st.image(
                resize_for_preview(image),
                caption=f"Input preview • original size {image.size}",
                width=PREVIEW_W
            )
        else:
            st.caption("No image loaded.")

    with p2:
        if task == "inpaint":
            if mask_upload is not None:
                st.image(
                    resize_for_preview(mask_upload),
                    caption="Uploaded mask preview",
                    width=PREVIEW_W
                )
            else:
                st.caption("No uploaded mask.")
        else:
            st.caption("Mask preview hidden (not in inpaint mode).")

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Draw canvas (ONLY in inpaint + draw mode)
# -----------------------------

drawn_mask = None

if task == "inpaint" and image is not None and mask_source == "draw":
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Draw Mask (full image size)</div>", unsafe_allow_html=True)
    st.caption("Your faded image is used as the background. White strokes define the repair area.")

    stroke_rgba = rgba_from_hex(brush_color, brush_opacity)
    w, h = image.size
    faded = make_faded_background(image)

    canvas_result = st_canvas(
        fill_color="rgba(255,255,255,0)",
        stroke_width=int(brush_size),
        stroke_color=stroke_rgba,
        background_image=faded,  # assuming community fix works
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

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Mask preview (ONLY in inpaint)
# -----------------------------

if task == "inpaint" and image is not None:
    with st.container():
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Mask Preview (final binary)</div>", unsafe_allow_html=True)

        preview_btn = st.button("Update Mask Preview")

        if "mask_preview" not in st.session_state:
            st.session_state["mask_preview"] = None

        if preview_btn:
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

        show_mask = st.session_state["mask_preview"]
        if show_mask is None:
            show_mask = Image.new("L", image.size, 0)

        st.image(
            resize_for_preview(show_mask),
            caption="Final mask preview (standard-sized)",
            width=PREVIEW_W
        )

        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Run output
# -----------------------------

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

    # Output card
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Output</div>", unsafe_allow_html=True)

    if task == "superres":
        with st.spinner("Running Swin2SR x4..."):
            try:
                out = enhancer.run_superres(image, mode=superres_mode)
            except TypeError:
                out = enhancer.run_superres(image)

        st.image(out, caption="Super-Resolution x4", width=OUTPUT_W)

    elif task == "inpaint":
        m = get_active_mask(
            image=image,
            mask_source=mask_source,
            mask_upload=mask_upload,
            drawn_mask=drawn_mask,
            invert=invert_mask
        )

        if m is None:
            st.error("Mask required for inpainting.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        inpaint_kwargs = {
            **base_kwargs,
            "use_lora": bool(use_lora),
            "lora_scale": float(lora_scale),
        }

        with st.spinner("Running inpaint..."):
            out = enhancer.run_inpaint(image, m, **inpaint_kwargs)

        st.image(out, caption="Inpainting result", width=OUTPUT_W)

    elif task == "denoise":
        with st.spinner("Running denoise..."):
            denoise_kwargs = {**base_kwargs, "strength": float(strength)}
            out = enhancer.run_denoise(image, **denoise_kwargs)
        st.image(out, caption=f"Denoise {strength}", width=FULL_W)

    elif task == "colorize":
        with st.spinner("Running colorization..."):
            out = enhancer.run_colorize(image)

        st.image(out, caption="Colorization result", width=512)

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Full input objects (optional reference)
# -----------------------------

with st.expander("Full Input Objects (original sizes)", expanded=False):
    c1, c2 = st.columns(2)

    with c1:
        if image is not None:
            st.image(image, caption=f"Image: {image.size}", width=FULL_W)
        else:
            st.caption("No image loaded.")

    with c2:
        if task == "inpaint":
            if mask_upload is not None:
                st.image(mask_upload, caption=f"Uploaded mask: {mask_upload.size}", width=FULL_W)
            elif drawn_mask is not None:
                st.image(drawn_mask, caption=f"Drawn mask: {drawn_mask.size}", width=FULL_W)
            else:
                st.caption("No mask loaded/drawn yet.")
        else:
            st.caption("Mask display hidden (not in inpaint mode).")
