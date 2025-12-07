# src/pipelines/colorize/pipeline.py
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from .model import UNet
from .utils import image_preprocessing, denormalize_ab
from .hf_weights import download_colorization_weights


@dataclass
class ColorizeResult:
    image: Image.Image


class UNetColorizationPipeline:
    """
    Lightweight non-diffusion colorizer.
    Task: grayscale/RGB -> colorized RGB using pretrained UNet.
    """

    def __init__(
        self,
        repo_id: str = "ayushshah/imagecolorization",
        weights_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.device = device

        if weights_path is None:
            weights_path = download_colorization_weights(repo_id=repo_id)

        state_dict = load_file(weights_path)

        self.model = UNet().to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, image: Image.Image, **kwargs) -> ColorizeResult:
        """
        kwargs ignored for compatibility with your enhancer signature.
        """
        if image is None:
            raise ValueError("Image is required for colorization.")

        # Ensure RGB input for preprocessing
        image = image.convert("RGB")

        L, L_norm = image_preprocessing(image, device=self.device)

        ab_pred = self.model(L_norm)
        ab = denormalize_ab(ab_pred)

        lab = torch.cat((L, ab), dim=1)

        # LAB -> RGB
        lab_t = lab.permute(0, 2, 3, 1)[0].detach().cpu().numpy().astype(np.float32)
        rgb = cv2.cvtColor(lab_t, cv2.COLOR_LAB2RGB)

        # rgb is float-like in [0,1] domain depending on OpenCV path;
        # enforce uint8 safely
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8) if rgb.max() <= 1.5 else np.clip(rgb, 0, 255).astype(np.uint8)

        out = Image.fromarray(rgb, mode="RGB")
        return ColorizeResult(image=out)
