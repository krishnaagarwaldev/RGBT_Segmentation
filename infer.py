"""Inference script for CSFNet-1 on single images or test sets with visual generation.

Usage:
  python infer.py --checkpoint runs/csfnet1_mfnet/best_model.pth --sample 01402D
  python infer.py --checkpoint runs/csfnet1_mfnet/best_model.pth --samples 00001D 00003N 01402D
"""
import os
import argparse
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

from configs.default_config import get_default_config
from models.csfnet import build_csfnet1
from utils.visualization import (
    colorize_mask,
    create_comparison_plot,
    denormalize_rgb,
    denormalize_thermal,
    CLASS_NAMES
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with CSFNet-1")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--dataset-root", type=str, default="ir_seg_dataset", help="Dataset root directory")
    parser.add_argument("--sample", type=str, default=None, help="Single sample stem (e.g., 01402D)")
    parser.add_argument("--samples", nargs="+", default=None, help="List of sample stems (e.g., 00001D 00003N)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output-dir", type=str, default="runs/inferences", help="Directory to save visual outputs")
    return parser.parse_args()


def run_inference_on_sample(
    sample_name: str,
    model: torch.nn.Module,
    dataset_root: str,
    output_dir: str,
    device: torch.device,
    cfg
):
    images_dir = os.path.join(dataset_root, "images")
    labels_dir = os.path.join(dataset_root, "labels")

    img_path = os.path.join(images_dir, f"{sample_name}.png")
    lbl_path = os.path.join(labels_dir, f"{sample_name}.png")

    if not os.path.exists(img_path):
        print(f"[Warning] Image not found: {img_path}")
        return

    # 1. Load 4-channel image
    with Image.open(img_path) as full_img:
        if full_img.mode != "RGBA":
            full_img = full_img.convert("RGBA")
        img_np = np.array(full_img)

    rgb_np = img_np[:, :, :3]
    th_np = img_np[:, :, 3]

    rgb_pil = Image.fromarray(rgb_np, mode="RGB")
    th_pil = Image.fromarray(th_np, mode="L")

    # Load GT mask if available
    gt_mask = None
    if os.path.exists(lbl_path):
        with Image.open(lbl_path) as lbl_img:
            gt_mask = np.array(lbl_img.convert("L"), dtype=np.int64)

    # 2. Normalize tensors
    rgb_t = TF.to_tensor(rgb_pil)
    rgb_t = TF.normalize(rgb_t, mean=list(cfg.dataset.rgb_mean), std=list(cfg.dataset.rgb_std))
    rgb_t = rgb_t.unsqueeze(0).to(device)

    th_t = TF.to_tensor(th_pil)
    th_t = TF.normalize(th_t, mean=list(cfg.dataset.thermal_mean), std=list(cfg.dataset.thermal_std))
    th_t = th_t.unsqueeze(0).to(device)

    # 3. Model forward
    with torch.no_grad():
        logits = model(rgb_t, th_t)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # 4. Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save raw integer mask PNG
    raw_mask_path = os.path.join(output_dir, f"{sample_name}_mask_raw.png")
    Image.fromarray(pred, mode="L").save(raw_mask_path)

    # Save colorized mask PNG
    colored_mask = colorize_mask(pred)
    colored_mask_path = os.path.join(output_dir, f"{sample_name}_mask_color.png")
    Image.fromarray(colored_mask).save(colored_mask_path)

    # Save comparison plot
    plot_path = os.path.join(output_dir, f"{sample_name}_comparison.png")
    dummy_gt = gt_mask if gt_mask is not None else np.zeros_like(pred)
    create_comparison_plot(
        rgb_img=rgb_np,
        thermal_img=th_np,
        gt_mask=dummy_gt,
        pred_mask=pred,
        save_path=plot_path,
        title=f"CSFNet-1 Prediction - Sample: {sample_name}"
    )

    detected_classes = [CLASS_NAMES[c] for c in np.unique(pred) if c > 0]
    print(f"Sample: {sample_name:<10} | Detected Classes: {detected_classes if detected_classes else 'None (Background only)'}")
    print(f"  -> Visual Comparison: {plot_path}")


def main():
    args = parse_args()
    cfg = get_default_config()
    device = torch.device(args.device)

    # Gather samples
    sample_list = []
    if args.sample:
        sample_list.append(args.sample)
    if args.samples:
        sample_list.extend(args.samples)
    if not sample_list:
        # Default to a few representative test day and night samples
        sample_list = ["00001D", "00003N", "01402D"]

    print("=" * 65)
    print("                CSFNet-1 RGBT INFERENCE")
    print("=" * 65)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print(f"Samples    : {sample_list}")
    print(f"Output Dir : {args.output_dir}")
    print("=" * 65)

    # Load Model
    model = build_csfnet1(num_classes=cfg.dataset.num_classes, dataset="MFNet").to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    for s_name in sample_list:
        run_inference_on_sample(s_name, model, args.dataset_root, args.output_dir, device, cfg)

    print("\nInference complete! All results saved to:", args.output_dir)


if __name__ == "__main__":
    main()
