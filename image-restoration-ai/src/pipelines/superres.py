from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

from ..utils.timing import timed  # if you have this already


@dataclass
class SuperResResult:
    image: Image.Image


class Swin2SRPipeline:
    """
    Two-mode Swin2SR x4 pipeline:
      - "Crystal clear (pretrained)" uses HF repo
      - "Smooth (fine-tuned)" uses local folder
    """

    MODES = ["Crystal clear (pretrained)", "Smooth (fine-tuned)"]

    def __init__(
        self,
        pretrained_id: str,
        finetuned_id: str,
        device: str = "cpu",
        mixed_precision: str = "no",
    ):
        self.device = device
        self.pretrained_id = pretrained_id
        self.finetuned_id = finetuned_id

        self.dtype = (
            torch.float16
            if device.startswith("cuda") and mixed_precision == "fp16"
            else torch.float32
        )

        with timed("Init Swin2SR processors"):
            self.processor_pre = AutoImageProcessor.from_pretrained(pretrained_id)
            # local fine-tuned processor
            self.processor_ft = AutoImageProcessor.from_pretrained(
                finetuned_id, local_files_only=True
            )

        with timed("Init Swin2SR models"):
            self.model_pre = Swin2SRForImageSuperResolution.from_pretrained(
                pretrained_id
            ).to(self.device)
            self.model_ft = Swin2SRForImageSuperResolution.from_pretrained(
                finetuned_id, local_files_only=True
            ).to(self.device)

        self.model_pre.eval()
        self.model_ft.eval()

        # Optional dtype casting (safe)
        if self.device.startswith("cuda") and self.dtype == torch.float16:
            try:
                self.model_pre.to(dtype=torch.float16)
                self.model_ft.to(dtype=torch.float16)
            except Exception:
                pass

        print(f"[SuperRes] device={self.device} dtype={self.dtype}")

    @torch.no_grad()
    def __call__(self, image: Image.Image, mode: str = "Crystal clear (pretrained)") -> SuperResResult:
        if image is None:
            raise ValueError("Image is required for super-resolution.")

        if mode == "Smooth (fine-tuned)":
            model = self.model_ft
            processor = self.processor_ft
        else:
            model = self.model_pre
            processor = self.processor_pre

        # Processor returns pixel_values
        inputs = processor(images=image, return_tensors="pt")

        # Move to device + dtype
        inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}

        # Forward
        outputs = model(**inputs)

        # Reconstruction -> PIL
        sr_tensor = outputs.reconstruction.squeeze().clamp(0, 1)
        sr_array = (
            sr_tensor.mul(255)
            .byte()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )
        sr_image = Image.fromarray(sr_array)

        return SuperResResult(image=sr_image)
