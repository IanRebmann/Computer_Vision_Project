import image_restoration_ai_ext.src.pipelines.colorize
from image_restoration_ai_ext.src.pipelines.colorize.colorization_object import ColorizationModel
from image_restoration_ai_ext.src.pipelines.inpaint.inpaint import SDInpaintPipeline
from image_restoration_ai_ext.src.pipelines.superres.superres import Swin2SRPipeline

import os
from dotenv import load_dotenv

# Load environment variables from the .env file (if present)
load_dotenv()

class Models:
    def __init__(
            self,
    ):
        self.inpainting_model = SDInpaintPipeline(model_id="sd-legacy/stable-diffusion-inpainting",
                                                  lora_dir=f"{os.getenv('ROOT_PATH')}/image_restoration_ai_ext/lora_inpainting")
        self.colorization_model = ColorizationModel()
        self.superres_model = Swin2SRPipeline(model_id="caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr") #eventually setup fine-tuned dir
        #self.state_dict = load_file(self.weights_init)
        #self.l_model = model_init(self.state_dict)

initialized_models = Models()
