import cv2
import torch

# The model expects a single L channel input of size 224x224, hence the image is center-cropped and resized
# The L channel values should be in the range [0, 1] hence it is normalized before being returned

def image_preprocessing(image):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    #if image is None or not os.path.exists(f'{image}'):
    #    return None
    try:

        h = image.shape[0]
        w = image.shape[1]

        module_h = h//224
        h = int((224*module_h))
        module_w = w//224
        w = int((224*module_w))

        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
        #print(f"{image.shape}")
        image = image.astype("float32") / 255
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # print(f"Lab image {lab_image.shape}\n")
        lab_tensor = torch.from_numpy(lab_image).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

        L = lab_tensor[:, 0:1, :, :]

    except Exception as e:
        print(f"Error {e} during image preprocessing when applying colorization.")
        return None

    return L, L / 100.0 # L + Normalized L

# The model outputs ab channels in the range [-1, 1]
# This functions uses linear scaling function to map the output back to the original ab range of [-128, 127]
def denormalize_ab(ab):
    ab = (ab+1)*255.0/2-128.0
    ab = torch.clamp(ab, -128, 127)
    return ab