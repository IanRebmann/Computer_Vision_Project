from dataclasses import dataclass
from PIL import Image
from pathlib import Path
import os
import torch
from ..utils.timing import timed

try:
    from diffusers import StableDiffusionInpaintPipeline
except Exception as e:
    StableDiffusionInpaintPipeline = None


@dataclass
class InpaintResult:
    image: Image.Image


class SDInpaintPipeline:
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        mixed_precision: str = "no",
        lora_dir: str | None = None,
    ):
        if StableDiffusionInpaintPipeline is None:
            raise ImportError("diffusers is not available or failed to import.")

        dtype = torch.float16 if (device.startswith("cuda") and mixed_precision == "fp16") else torch.float32

        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HF_HUB_TOKEN")
        )

        with timed(f"Load StableDiffusionInpaintPipeline: {model_id}"):
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                token=token,
            )

        with timed("Move inpaint pipeline to device"):
            self.pipe.to(device)

        self.device = device

        print(f"[Inpaint] device={device} dtype={self.pipe.unet.dtype}")
        print(f"[Inpaint] torch.cuda.is_available()={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[Inpaint] current GPU={torch.cuda.get_device_name(0)}")

        if device.startswith("cuda"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass

        # --- LoRA support ---
        self.lora_loaded = False
        self.lora_dir = None

        if lora_dir:
            p = Path(lora_dir)
            if p.exists() and p.is_dir():
                with timed(f"Load inpaint LoRA from {lora_dir}"):
                    self.pipe.load_lora_weights(str(p))
                self.lora_loaded = True
                self.lora_dir = str(p)

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        guidance_scale: float = 6.0,
        num_inference_steps: int = 28,
        seed: int | None = None,
        use_lora: bool = True,
        lora_scale: float = 1.0,
    ) -> InpaintResult:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # If LoRA is loaded and not fused, control scale
        if self.lora_loaded and hasattr(self.pipe, "set_adapters"):
            try:
                if use_lora:
                    self.pipe.set_adapters(["default"], [float(lora_scale)])
                else:
                    self.pipe.set_adapters(["default"], [0.0])
            except Exception:
                pass

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=image,
            mask_image=mask,
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_inference_steps),
            generator=generator,
        )

        return InpaintResult(image=out.images[0])
