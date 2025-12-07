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
    def __call__(self, image: Image.Image, **kwargs):
        # Ensure RGB input
        image = image.convert("RGB")

        # ✅ Convert PIL -> numpy RGB
        image_np = np.array(image)

        # Now preprocessing expects numpy
        L, L_norm = image_preprocessing(image_np, device=self.device)

        ab_pred = self.model(L_norm)
        ab = denormalize_ab(ab_pred)

        lab = torch.cat((L, ab), dim=1)
        lab = lab.permute(0, 2, 3, 1)

        image_lab = lab[0].detach().cpu().numpy()
        rgb = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)

        # ✅ Convert to PIL
        rgb_u8 = np.clip(rgb * 255 if rgb.max() <= 1.0 else rgb, 0, 255).astype(np.uint8)
        out = Image.fromarray(rgb_u8)

        return ColorizeResult(image=out)