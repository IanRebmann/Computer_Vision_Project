import os
from PIL import Image, ImageOps
import numpy as np
from kornia.color import rgb_to_lab
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The model expects a single L channel input of size 224x224, hence the image is center-cropped and resized
# The L channel values should be in the range [0, 1] hence it is normalized before being returned
# You can change the notebook code to make the model make predictions on a batch of images instead of a single image
def load_image(image):
    if image is None or not os.path.exists(f'{image}'):
        return None
    try:
        img = Image.open(f'{image}').convert('RGB')
        side = min(img.size)
        img = ImageOps.fit(img, (side, side), centering=(0.5, 0.5))
        img = img.resize((224, 224), Image.Resampling.BICUBIC)
    except Exception as e:
        return None

    img = np.array(img) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    lab_tensor = rgb_to_lab(img)

    L = lab_tensor[:, 0:1, :, :]
    return L, L / 100.0

# The model outputs ab channels in the range [-1, 1]
# This functions uses linear scaling function to map the output back to the original ab range of [-128, 127]
def denormalize_ab(ab):
    ab = (ab+1)*255.0/2-128.0
    ab = torch.clamp(ab, -128, 127)
    return ab