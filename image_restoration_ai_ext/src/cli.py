import argparse
from pathlib import Path

from PIL import Image

from .config import load_config
from .enhancer import ImageEnhancer
from .utils.io import load_image, save_image

def build_parser():
    p = argparse.ArgumentParser("Image Restoration AI")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--task", choices=["denoise", "superres", "colorize", "inpaint"], required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mask", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--negative_prompt", default=None)
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p

def main():
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    enhancer = ImageEnhancer(cfg)

    img = load_image(args.input)

    kwargs = {}
    for k in ["prompt", "negative_prompt", "strength", "guidance_scale", "seed"]:
        v = getattr(args, k)
        if v is not None:
            kwargs[k] = v
    if args.steps is not None:
        kwargs["num_inference_steps"] = args.steps

    if args.task == "denoise":
        out = enhancer.run_denoise(img, **kwargs)

    elif args.task == "superres":
        out = enhancer.run_superres(img)

    elif args.task == "colorize":
        out = enhancer.run_colorize(img, **kwargs)

    elif args.task == "inpaint":
        if not args.mask:
            raise SystemExit("--mask is required for inpaint task.")
        mask = Image.open(args.mask).convert("RGB")
        out = enhancer.run_inpaint(img, mask, **kwargs)

    else:
        raise SystemExit("Unknown task.")

    save_image(out, args.output)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
