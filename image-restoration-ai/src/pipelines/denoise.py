from dataclasses import dataclass
from PIL import Image
import torch

try:
    from diffusers import StableDiffusionImg2ImgPipeline
except Exception:
    StableDiffusionImg2ImgPipeline = None


@dataclass
class DenoiseResult:
    image: Image.Image


class SDDenoiseImg2ImgPipeline:
    def __init__(self, model_id: str, device: str = "cpu", mixed_precision: str = "no"):
        if StableDiffusionImg2ImgPipeline is None:
            raise ImportError("diffusers is not available or failed to import.")

        dtype = torch.float16 if (device.startswith("cuda") and mixed_precision == "fp16") else torch.float32

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )
        self.pipe.to(device)
        self.device = device

        # Optional local-only choice:
        try:
            self.pipe.safety_checker = None
        except Exception:
            pass

        # Memory helpers
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass

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
        strength: float = 0.25,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 20,
        seed: int | None = None,
    ) -> DenoiseResult:

        image = image.convert("RGB")
        negative_prompt = negative_prompt or ""

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        out = self.pipe(
            prompt=prompt or "clean, realistic photo, no text, no watermark",
            negative_prompt=negative_prompt,
            image=image,
            strength=float(strength),
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_inference_steps),
            generator=generator,
        )
        return DenoiseResult(image=out.images[0])
