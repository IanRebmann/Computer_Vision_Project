import numpy as np
from PIL import Image

def to_gray(img: Image.Image) -> Image.Image:
    return img.convert("L").convert("RGB")

def resize_to_multiple(img: Image.Image, multiple: int = 8) -> Image.Image:
    w, h = img.size
    nw = (w // multiple) * multiple
    nh = (h // multiple) * multiple
    if nw == 0 or nh == 0:
        return img
    if (nw, nh) == (w, h):
        return img
    return img.resize((nw, nh), Image.LANCZOS)

def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)

def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))
