# https://huggingface.co/ayushshah/imagecolorization
# https://colab.research.google.com/drive/1JYbSLtDuFSw2NYe-YW-kZHLNkt4-v7jd#scrollTo=MAIfEjICqnXV
import cv2

from image_restoration_ai_ext.src.pipelines.colorize.model import UNet
from image_restoration_ai_ext.src.pipelines.colorize.colorization_utils import image_preprocessing, denormalize_ab
from image_restoration_ai_ext.src.pipelines.colorize.colorization_model import colorization_execute
from safetensors.torch import load_file
import torch

def model_init(state_dict):
    try:
        print("[Colorization] Loading colorization model.")
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        model = UNet().to(DEVICE)
        model.load_state_dict(state_dict)
        _ = model.eval()
        return model
    except Exception as e:
        print(f"Error during model initialization {e}.")

class ColorizationModel():
    def __init__(
            self,
    ):
        self.weights_init = colorization_execute()
        self.state_dict = load_file(self.weights_init)
        self.l_model = model_init(self.state_dict)

    def colorize(self, image):
        try:
            L, L_normalized = image_preprocessing(image)

            with torch.no_grad():
                ab_pred = self.l_model(L_normalized)

            ab = denormalize_ab(ab_pred)
            lab = torch.cat((L, ab), dim=1)

            # Convert LAB back to RGB
            lab_t = lab.permute(0, 2, 3, 1)
            image_lab = lab_t[0].numpy()
            lab_image = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)
            return lab_image
        except Exception as e:
            print(f"Error during image colorization {e}.")

    def display(self, image):
        window_name = 'colorized'
        cv2.imshow(window_name, image)
        cv2.waitKey(0)