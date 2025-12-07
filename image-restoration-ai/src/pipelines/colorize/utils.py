# src/pipelines/colorize/utils.py
import cv2
import numpy as np
import torch
from PIL import Image

W = 224
H = 224

def _to_rgb_np(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        # assume already RGB
        return image
    raise TypeError("image must be PIL.Image or numpy array")

def image_preprocessing(image, device: str):
    """
    Returns:
      L (1,1,H,W) in LAB scale
      L_normalized (1,1,H,W) scaled to [0,1] by /100
    """
    img = _to_rgb_np(image)

    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32") / 255.0

    lab_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_tensor = torch.from_numpy(lab_image).permute(2, 0, 1).unsqueeze(0).float().to(device)

    L = lab_tensor[:, 0:1, :, :]  # L channel in [0,100]
    return L, L / 100.0

def denormalize_ab(ab: torch.Tensor):
    """
    Model outputs ab in [-1,1]. Convert back to LAB ab range approx [-128,127]
    """
    ab = (ab + 1) * 255.0 / 2 - 128.0
    ab = torch.clamp(ab, -128, 127)
    return ab
