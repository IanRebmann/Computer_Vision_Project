from dataclasses import dataclass
from PIL import Image
import torch

from .config import AppConfig
from .pipelines.superres import Swin2SRPipeline
from .pipelines.inpaint import SDInpaintPipeline
from .pipelines.denoise import SDDenoiseImg2ImgPipeline
from .pipelines.colorize.colorize import UNetColorizationPipeline
from .utils.timing import timed

@dataclass
class EnhanceOutputs:
    denoised: Image.Image | None = None
    superres: Image.Image | None = None
    colorized: Image.Image | None = None
    inpainted: Image.Image | None = None

class ImageEnhancer:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        device = cfg.device
        default_mp = "fp16" if str(device).startswith("cuda") else "no"
        mp = cfg.runtime.get("mixed_precision", default_mp)

        print(f"[Enhancer] device={device} mixed_precision={mp}")
        print(f"[Enhancer] cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[Enhancer] gpu={torch.cuda.get_device_name(0)}")

        with timed("Init Swin2SR superres Pipeline (with two-mode)"):
            self.superres = Swin2SRPipeline(
                pretrained_id=cfg.models["superres_pretrained_id"],
                finetuned_id=cfg.models["superres_finetuned_id"],
                device=device,
                mixed_precision=mp,
            )

        with timed("Init SD Inpaint Pipeline (with optional LoRA)"):
            self.inpaint = SDInpaintPipeline(
                cfg.models["inpaint_model_id"],
                device=device,
                mixed_precision=mp,
                lora_dir=cfg.models.get("inpaint_lora_dir", None),
            )

        #with timed("Init SD Denoise Img2Img Pipeline"):
        #    self.denoise = SDDenoiseImg2ImgPipeline(
        #        cfg.models["img2img_model_id"], device=device, mixed_precision=mp
        #    )

        with timed("Init UNet Colorize Pipeline"):
            self.colorize = UNetColorizationPipeline(
                repo_id=cfg.models.get("colorize_repo_id", "ayushshah/imagecolorization"),
                weights_path=cfg.models.get("colorize_weights_path", None),
                device=device,
            )

        # Optional: warm-up to avoid first-click lag
        if str(device).startswith("cuda"):
            with timed("Warm-up (noop GPU sync)"):
                torch.cuda.synchronize()

    def run_denoise(self, image: Image.Image, **kwargs) -> Image.Image:
        defaults = self.cfg.defaults.get("denoise", {})
        params = {**defaults, **kwargs}
        with timed("Denoise inference"):
            return self.denoise(image=image, **params).image

    def run_superres(self, image: Image.Image, mode: str | None = None, **kwargs) -> Image.Image:
        defaults = self.cfg.defaults.get("superres", {})
        if mode is None:
            mode = defaults.get("mode", "Crystal clear (pretrained)")
        return self.superres(image=image, mode=mode).image

    def run_colorize(self, image: Image.Image, **kwargs) -> Image.Image:
        defaults = self.cfg.defaults.get("colorize", {})
        params = {**defaults, **kwargs}
        with timed("Colorize inference"):
            return self.colorize(image=image, **params).image

    def run_inpaint(self, image, mask, **kwargs):
        defaults = self.cfg.defaults.get("inpaint", {})
        params = {**defaults, **kwargs}
        params.pop("strength", None)  # defensive
        print("[Inpaint] run_inpaint params:", params)
        #with timed("Inpaint inference"):
        return self.inpaint(image=image, mask=mask, **params).image
        
    def run_all_optional(
        self,
        image: Image.Image,
        do_denoise=False,
        do_superres=False,
        do_colorize=False,
        do_inpaint=False,
        mask: Image.Image | None = None,
        superres_mode: str | None = None,
        **kwargs
    ) -> EnhanceOutputs:
        out = EnhanceOutputs()

        current = image

        if do_denoise:
            current = out.denoised = self.run_denoise(current, **kwargs)

        if do_colorize:
            current = out.colorized = self.run_colorize(current, **kwargs)

        if do_superres:
            current = out.superres = self.run_superres(current, mode=superres_mode)

        if do_inpaint:
            if mask is None:
                raise ValueError("Mask is required for inpainting.")
            current = out.inpainted = self.run_inpaint(current, mask, **kwargs)

        return out
