"""Evaluation script for CSFNet-1 on validation or test sets (overall, day, night).

Usage:
  python evaluate.py --checkpoint runs/csfnet1_mfnet/best_model.pth --split test
  python evaluate.py --checkpoint runs/csfnet1_mfnet/best_model.pth --split test_day
  python evaluate.py --checkpoint runs/csfnet1_mfnet/best_model.pth --split test_night
"""
import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.default_config import get_default_config
from data.dataset import RGBTDataset
from data.transforms import get_val_transforms
from models.csfnet import build_csfnet1
from utils.metrics import SegmentationMetrics
from utils.visualization import (
    create_comparison_plot,
    denormalize_rgb,
    denormalize_thermal
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CSFNet-1 on RGBT dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--dataset-root", type=str, default="ir_seg_dataset", help="Path to dataset root")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test", "test_day", "test_night"],
                        help="Split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--backbone", type=str, default="auto",
                        choices=["auto", "STDCNet813", "STDCNet1446"],
                        help="Backbone architecture (auto-detects from checkpoint by default)")
    parser.add_argument("--save-visuals", action="store_true", default=True, help="Save prediction visualizations")
    parser.add_argument("--max-visuals", type=int, default=15, help="Maximum visual plots to save")
    parser.add_argument("--output-dir", type=str, default="runs/eval_results", help="Directory to save evaluation results")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_default_config()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    split_file_map = {
        "val": os.path.join(args.dataset_root, "val.txt"),
        "test": os.path.join(args.dataset_root, "test.txt"),
        "test_day": os.path.join(args.dataset_root, "test_day.txt"),
        "test_night": os.path.join(args.dataset_root, "test_night.txt")
    }
    split_file = split_file_map[args.split]

    print("=" * 65)
    print(f"        CSFNet-1 EVALUATION ON SPLIT: '{args.split.upper()}'")
    print("=" * 65)
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Split File  : {split_file}")
    print(f"Device      : {device}")
    print("=" * 65)

    # 1. Dataset & DataLoader
    val_transforms = get_val_transforms(
        rgb_mean=cfg.dataset.rgb_mean,
        rgb_std=cfg.dataset.rgb_std,
        thermal_mean=cfg.dataset.thermal_mean,
        thermal_std=cfg.dataset.thermal_std
    )
    dataset = RGBTDataset(
        dataset_root=args.dataset_root,
        split_file=split_file,
        transforms=val_transforms,
        use_unique_stems=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda")
    )
    print(f"Total Evaluation Samples: {len(dataset)}")

    # 2. Model & Checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    if args.backbone == "auto":
        is_stdc2 = any("encoder.features_rgb.13" in k for k in state_dict.keys())
        backbone_name = "STDCNet1446" if is_stdc2 else "STDCNet813"
    else:
        backbone_name = args.backbone

    from models.csfnet import build_csfnet
    model = build_csfnet(backbone=backbone_name, pretrained=False, num_classes=cfg.dataset.num_classes, dataset="MFNet").to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Evaluation Loop
    metrics = SegmentationMetrics(num_classes=cfg.dataset.num_classes, class_names=cfg.dataset.class_names)
    visuals_saved = 0
    vis_dir = os.path.join(args.output_dir, f"visuals_{args.split}")
    if args.save_visuals:
        os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating [{args.split}]"):
            rgb = batch["rgb"].to(device, non_blocking=True)
            thermal = batch["thermal"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            logits = model(rgb, thermal)
            preds = torch.argmax(logits, dim=1)
            metrics.update(preds, mask)

            if args.save_visuals and visuals_saved < args.max_visuals:
                for b_idx in range(len(batch["name"])):
                    if visuals_saved >= args.max_visuals:
                        break
                    s_name = batch["name"][b_idx]
                    s_rgb = denormalize_rgb(rgb[b_idx].cpu().numpy(), cfg.dataset.rgb_mean, cfg.dataset.rgb_std)
                    s_th = denormalize_thermal(thermal[b_idx].cpu().numpy(), cfg.dataset.thermal_mean, cfg.dataset.thermal_std)
                    s_gt = mask[b_idx].cpu().numpy()
                    s_pred = preds[b_idx].cpu().numpy()

                    out_img_path = os.path.join(vis_dir, f"{s_name}_eval.png")
                    create_comparison_plot(
                        s_rgb, s_th, s_gt, s_pred,
                        save_path=out_img_path,
                        title=f"Sample: {s_name} ({args.split})"
                    )
                    visuals_saved += 1

    # 4. Results
    scores = metrics.compute()
    summary_table = metrics.get_summary_table()
    print("\n" + summary_table)

    # Save to JSON
    results_path = os.path.join(args.output_dir, f"metrics_{args.split}.json")
    save_dict = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "pixel_accuracy": scores["pixel_accuracy"],
        "mIoU": scores["mIoU"],
        "mIoU_fg": scores["mIoU_fg"],
        "class_iou": scores["class_iou"],
        "confusion_matrix": scores["confusion_matrix"].tolist()
    }
    with open(results_path, "w") as f:
        json.dump(save_dict, f, indent=2)

    print(f"\nSaved metrics JSON to: {results_path}")
    if args.save_visuals:
        print(f"Saved {visuals_saved} visual comparisons to: {vis_dir}")


if __name__ == "__main__":
    main()
