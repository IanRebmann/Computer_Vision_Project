import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def pil_to_arr(img):
    return np.array(img).astype(np.float32)

def compute_metrics(pred: Image.Image, gt: Image.Image):
    p = pil_to_arr(pred)
    g = pil_to_arr(gt)
    psnr = peak_signal_noise_ratio(g, p, data_range=255)
    ssim = structural_similarity(g, p, channel_axis=2, data_range=255)
    return psnr, ssim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    preds = sorted(pred_dir.glob("*.png"))
    if not preds:
        raise SystemExit("No predictions found.")

    all_psnr, all_ssim = [], []

    for p in preds:
        gt_path = gt_dir / p.name
        if not gt_path.exists():
            continue
        pred = Image.open(p).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        psnr, ssim = compute_metrics(pred, gt)
        all_psnr.append(psnr)
        all_ssim.append(ssim)

    print(f"Count: {len(all_psnr)}")
    if all_psnr:
        print(f"PSNR mean: {sum(all_psnr)/len(all_psnr):.3f}")
        print(f"SSIM mean: {sum(all_ssim)/len(all_ssim):.3f}")

if __name__ == "__main__":
    main()
