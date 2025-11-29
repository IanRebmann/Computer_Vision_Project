# https://huggingface.co/ayushshah/imagecolorization
# https://colab.research.google.com/drive/1JYbSLtDuFSw2NYe-YW-kZHLNkt4-v7jd#scrollTo=MAIfEjICqnXV

from model import UNet
from colorization_utils import load_image, denormalize_ab
from colorization_model import weights_path
from safetensors.torch import load_file
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(DEVICE)
state_dict = load_file(weights_path)
model.load_state_dict(state_dict)
_ = model.eval()

# Note: Upload the image file to the '/contents/' directory from the left sidebar

from kornia.color import lab_to_rgb
import matplotlib.pyplot as plt

# If the path is '/content/image.jpg', just write 'image.jpg' as the input
img_file = "/home/alberto/Documents/MSAAI/CV/final_project/Computer_Vision_Project/carlos/test.jpg"

L, L_normalized = load_image(img_file)

with torch.no_grad():
    ab_pred = model(L_normalized)

ab = denormalize_ab(ab_pred)
lab = torch.cat((L, ab), dim=1)
rgb = lab_to_rgb(lab)

# Show L image and colorized rgb
fig, axes = plt.subplots(1, 2)
axes[0].imshow(L[0, 0].cpu().numpy(), cmap='gray', aspect='equal')
axes[0].set_title('Input')
axes[0].axis('off')
axes[1].imshow(rgb[0].permute(1, 2, 0).cpu().numpy(), aspect='equal')
axes[1].set_title('Prediction')
axes[1].axis('off')

plt.subplots_adjust(wspace=0, hspace=0, left=0.129)
plt.show()

