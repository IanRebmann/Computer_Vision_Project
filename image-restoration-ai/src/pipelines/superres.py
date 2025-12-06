from dataclasses import dataclass
from PIL import Image
import torch
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

@dataclass
class SuperResResult:
    image: Image.Image

class Swin2SRPipeline:
    def __init__(self, model_id: str, device: str = "cpu"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = Swin2SRForImageSuperResolution.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> SuperResResult:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # outputs.reconstruction is (1, 3, H, W)
        rec = outputs.reconstruction.clamp(0, 1)
        rec = rec[0].permute(1, 2, 0).cpu().numpy()
        rec = (rec * 255.0).round().astype("uint8")
        out_img = Image.fromarray(rec)
        return SuperResResult(image=out_img)
