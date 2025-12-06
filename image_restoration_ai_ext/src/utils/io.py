from PIL import Image
from pathlib import Path

def load_image(path: str | Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

def save_image(img: Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))
