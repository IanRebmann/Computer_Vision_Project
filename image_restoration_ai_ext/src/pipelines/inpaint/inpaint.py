from dataclasses import dataclass

import numpy as np
from PIL import Image
from pathlib import Path
import os
import torch
import time
from image_restoration_ai_ext.src.utils.timing import timed

try:
    from diffusers import StableDiffusionInpaintPipeline
except Exception:
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
        print("Initializing SDInpaintPipeline...")
        
        if StableDiffusionInpaintPipeline is None:
            raise ImportError("diffusers is not available or failed to import.")

        self.device = device

        # dtype selection
        use_fp16 = device.startswith("cuda") and mixed_precision == "fp16"
        dtype = torch.float16 if use_fp16 else torch.float32

        #token = (
        #    os.environ.get("HF_TOKEN")
        #    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        #    or os.environ.get("HF_HUB_TOKEN")
        #)

        with timed(f"Load StableDiffusionInpaintPipeline: {model_id}"):
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                dtype=dtype,
                #token=token,
            )

        with timed("Move inpaint pipeline to device"):
            self.pipe.to(device)

        # --- Speed knobs for CUDA ---
        if device.startswith("cuda"):
            # TF32 can help a bit
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # xFormers
            #try:
            #    self.pipe.enable_xformers_memory_efficient_attention()
            #    print("[Inpaint] xFormers enabled ✅")
            #except Exception as e:
            #    print("[Inpaint] xFormers not enabled ❌", e)

            # VAE slicing is usually safe
            #try:
            #    self.pipe.enable_vae_slicing()
            #except Exception:
            #    pass

            # IMPORTANT:
            # Do NOT enable attention slicing on a 4090 unless you are OOM
            # self.pipe.enable_attention_slicing()

        def _dev(m):
            try:
                return next(m.parameters()).device
            except StopIteration:
                return "no-params"

        print("[Inpaint] UNet device:", _dev(self.pipe.unet), flush=True)
        print("[Inpaint] VAE device:", _dev(self.pipe.vae), flush=True)
        print("[Inpaint] TextEnc device:", _dev(self.pipe.text_encoder), flush=True)

        print(f"[Inpaint] device={device} dtype={self.pipe.unet.dtype}")
        print(f"[Inpaint] torch.cuda.is_available()={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[Inpaint] current GPU={torch.cuda.get_device_name(0)}")

        # --- LoRA support ---
        self.lora_loaded = False
        self.lora_dir = None
        self.adapter_name = None

        if lora_dir:
            p = Path(lora_dir)
            if p.exists() and p.is_dir():
                self.adapter_name = "inpaint_lora"
                with timed(f"Load inpaint LoRA from {lora_dir}"):
                    # Explicit adapter name for reliable scaling control
                    self.pipe.load_lora_weights(str(p), adapter_name=self.adapter_name)
                self.lora_loaded = True
                self.lora_dir = str(p)

                # If you *always* want LoRA on and don't need runtime scaling,
                # you can fuse for speed:
                # try:
                #     self.pipe.fuse_lora(adapter_names=[self.adapter_name])
                #     print("[Inpaint] LoRA fused ✅")
                # except Exception:
                #     pass

    @torch.no_grad()
    def __call__(
        self,
        image: np.ndarray,
        mask: Image.Image,
        prompt: str = "",
        negative_prompt: str = "",
        guidance_scale: float = 6.0,
        num_inference_steps: int = 5,
        seed: int | None = None,
        use_lora: bool = True,
        lora_scale: float = 1.0,
    ) -> InpaintResult:

        params = {
            "guidance_scale": float(guidance_scale),
            "num_inference_steps": int(num_inference_steps),
            "use_lora": bool(use_lora),
            "lora_scale": float(lora_scale),
            "seed": seed,
            "image_size": getattr(image, "size", None),
            "mask_size": getattr(mask, "size", None),
        }
        print("[Inpaint] __call__ params:", params, flush=True)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Apply LoRA scale if supported
        if self.lora_loaded and hasattr(self.pipe, "set_adapters") and self.adapter_name:
            try:
                scale = float(lora_scale) if use_lora else 0.0
                self.pipe.set_adapters([self.adapter_name], [scale])
            except Exception:
                pass

        t0 = time.perf_counter()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"[GPU before] free={free/1e9:.2f} GB")

        with timed("Inpaint inference (diffusers)"):
            out = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                image=image,
                mask_image=mask,
                guidance_scale=float(guidance_scale),
                num_inference_steps=int(num_inference_steps),
                generator=generator,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            print(f"[GPU after ] free={free/1e9:.2f} GB")

        print(f"[Inpaint] inference wall time: {time.perf_counter() - t0:.2f}s")

        return InpaintResult(image=out.images[0])
