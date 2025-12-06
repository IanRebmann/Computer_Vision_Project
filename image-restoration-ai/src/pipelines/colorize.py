from dataclasses import dataclass
from PIL import Image
import torch

from ..utils.image_ops import to_gray

try:
    from diffusers import StableDiffusionImg2ImgPipeline
except Exception:
    StableDiffusionImg2ImgPipeline = None

@dataclass
class ColorizeResult:
    image: Image.Image

class SDColorizeImg2ImgPipeline:
    def __init__(self, model_id: str, device: str = "cpu", mixed_precision: str = "no"):
        if StableDiffusionImg2ImgPipeline is None:
            raise ImportError("diffusers is not available or failed to import.")

        dtype = torch.float16 if (device.startswith("cuda") and mixed_precision == "fp16") else torch.float32
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
        self.pipe.to(device)
        self.device = device

        if device.startswith("cuda"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.35,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 20,
        seed: int | None = None,
        force_grayscale_input: bool = True,
    ) -> ColorizeResult:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        base = to_gray(image) if force_grayscale_input else image

        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=base,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        return ColorizeResult(image=out.images[0])
