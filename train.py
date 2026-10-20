"""Main training script for CSFNet-1 (STDC1) RGBT Semantic Segmentation.

Features:
- Dual-modal RGBT loading (RGB + Thermal).
- Automatic Mixed Precision (AMP) for RTX 3050 6GB.
- Real-time progress bar (tqdm) with epoch-by-epoch tracking.
- Automatic metric logging to CSV and JSON.
- Live plot updates: loss curves, mIoU curves, and training dashboard.
- Periodic validation visual comparisons (RGB, Thermal, GT, Pred).
- Best model checkpointing & Early Stopping.
"""
import os
import sys
import time
import json
import csv
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.default_config import get_default_config, Config
from data.dataset import RGBTDataset
from data.transforms import get_train_transforms, get_val_transforms
from models.csfnet import build_csfnet, build_csfnet1
from models.losses import build_loss
from utils.metrics import SegmentationMetrics
from utils.checkpoint import CheckpointManager, EarlyStopping
from utils.visualization import (
    create_comparison_plot,
    plot_training_history,
    denormalize_rgb,
    denormalize_thermal
)


def set_seed(seed: int = 42):
    """Ensure deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train CSFNet on ir_seg_dataset")
    parser.add_argument("--backbone", type=str, default="STDCNet1446", choices=["STDCNet813", "STDCNet1446"],
                        help="STDC backbone architecture (STDCNet813 for CSFNet-1, STDCNet1446 for CSFNet-2)")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Load ImageNet-pretrained backbone weights")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Train backbone from scratch")
    parser.add_argument("--pretrained-path", type=str, default=None, help="Custom path to pretrained backbone tar/pth")
    parser.add_argument("--dataset-root", type=str, default="ir_seg_dataset", help="Path to dataset root")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=4, help="Validation batch size")
    parser.add_argument("--epochs", type=int, default=150, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay")
    parser.add_argument("--loss", type=str, default="combined_ce_dice",
                        choices=["ce", "weighted_ce", "focal", "dice", "combined_ce_dice", "combined_focal_dice"],
                        help="Loss function")
    parser.add_argument("--weight-strategy", type=str, default="enet_log",
                        choices=["none", "enet_log", "median_frequency", "power_smoothed", "inverse_frequency"],
                        help="Class weighting strategy for CE")
    parser.add_argument("--dice-weight", type=float, default=0.5, help="Dice weight multiplier")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience")
    parser.add_argument("--save-dir", type=str, default="runs/csfnet2_pretrained", help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-visuals-every", type=int, default=5, help="Save validation visual plots every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_default_config()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    visuals_dir = os.path.join(args.save_dir, "visualizations")
    os.makedirs(visuals_dir, exist_ok=True)

    arch_title = "CSFNet-2 (STDC2)" if "1446" in args.backbone else "CSFNet-1 (STDC1)"
    print("=" * 65)
    print(f"      {arch_title} RGBT SEMANTIC SEGMENTATION TRAINING")
    print("=" * 65)
    print(f"Device               : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Backbone             : {args.backbone} (Pretrained: {args.pretrained})")
    print(f"Mixed Precision (AMP): {args.amp}")
    print(f"Batch Size           : {args.batch_size} (Train), {args.val_batch_size} (Val)")
    print(f"Epochs               : {args.epochs}")
    print(f"Learning Rate        : {args.lr}")
    print(f"Loss Function        : {args.loss} (weight strategy: {args.weight_strategy})")
    print(f"Output Directory     : {args.save_dir}")
    print("=" * 65)

    # 1. Datasets and Loaders
    train_transforms = get_train_transforms(
        target_size=(cfg.dataset.image_height, cfg.dataset.image_width),
        hflip_p=cfg.augmentation.horizontal_flip_prob,
        scale_range=cfg.augmentation.random_scale_range,
        rgb_mean=cfg.dataset.rgb_mean,
        rgb_std=cfg.dataset.rgb_std,
        thermal_mean=cfg.dataset.thermal_mean,
        thermal_std=cfg.dataset.thermal_std
    )
    val_transforms = get_val_transforms(
        rgb_mean=cfg.dataset.rgb_mean,
        rgb_std=cfg.dataset.rgb_std,
        thermal_mean=cfg.dataset.thermal_mean,
        thermal_std=cfg.dataset.thermal_std
    )

    train_dataset = RGBTDataset(
        dataset_root=args.dataset_root,
        split_file=os.path.join(args.dataset_root, "train.txt"),
        transforms=train_transforms,
        use_unique_stems=True  # 784 unique base images with dynamic augmentation
    )
    val_dataset = RGBTDataset(
        dataset_root=args.dataset_root,
        split_file=os.path.join(args.dataset_root, "val.txt"),
        transforms=val_transforms,
        use_unique_stems=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda")
    )

    print(f"Train Samples: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f"Val Samples  : {len(val_dataset)} ({len(val_loader)} batches)")

    # 2. Model
    model = build_csfnet(
        backbone=args.backbone,
        pretrained=args.pretrained,
        pretrained_path=args.pretrained_path,
        num_classes=cfg.dataset.num_classes,
        dataset="MFNet"
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model Architecture   : {arch_title}, {total_params:.2f}M Parameters")

    # 3. Loss, Optimizer, Scaler, Scheduler
    criterion = build_loss(
        loss_type=args.loss,
        weight_strategy=args.weight_strategy,
        dice_weight=args.dice_weight,
        device=args.device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Robust AMP GradScaler setup compatible with all PyTorch 2.x versions
    scaler = None
    if args.amp and args.device == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler()

    # 4. Checkpoints, Early Stopping, Metrics
    ckpt_mgr = CheckpointManager(save_dir=args.save_dir)
    early_stopping = EarlyStopping(patience=args.patience, mode="max")
    val_metrics = SegmentationMetrics(num_classes=cfg.dataset.num_classes, class_names=cfg.dataset.class_names)

    # History logs
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_pixel_acc": [],
        "val_mIoU": [],
        "val_mIoU_fg": [],
        "lr": []
    }
    csv_log_path = os.path.join(args.save_dir, "training_metrics.csv")
    json_log_path = os.path.join(args.save_dir, "training_metrics.json")

    start_epoch = 1
    best_mIoU = 0.0

    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        ckpt_data = ckpt_mgr.load(args.resume, model=model, optimizer=optimizer, scaler=scaler, device=args.device)
        start_epoch = ckpt_data.get("epoch", 0) + 1
        best_mIoU = ckpt_data.get("metrics", {}).get("mIoU", 0.0)
        early_stopping.best_score = best_mIoU
        print(f"  -> Resumed at Epoch {start_epoch}, previous best mIoU: {best_mIoU*100:.2f}%")

        if os.path.exists(json_log_path):
            with open(json_log_path, "r") as f:
                history = json.load(f)
    else:
        with open(csv_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "pixel_acc", "mIoU_all", "mIoU_fg", "lr"])

    # Align scheduler if resumed
    if start_epoch > 1:
        scheduler.last_epoch = start_epoch - 1

    # 5. Training Loop
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs:03d} [Train]", leave=False)
        for batch in pbar:
            rgb = batch["rgb"].to(device, non_blocking=True)
            thermal = batch["thermal"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                # AMP Autocast
                try:
                    with torch.amp.autocast("cuda"):
                        logits = model(rgb, thermal)
                        loss = criterion(logits, mask)
                except AttributeError:
                    with torch.cuda.amp.autocast():
                        logits = model(rgb, thermal)
                        loss = criterion(logits, mask)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(rgb, thermal)
                loss = criterion(logits, mask)
                loss.backward()
                optimizer.step()

            running_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # 6. Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_metrics.reset()

        sample_visual_saved = False

        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(device, non_blocking=True)
                thermal = batch["thermal"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)

                if scaler is not None:
                    try:
                        with torch.amp.autocast("cuda"):
                            logits = model(rgb, thermal)
                            loss = criterion(logits, mask)
                    except AttributeError:
                        with torch.cuda.amp.autocast():
                            logits = model(rgb, thermal)
                            loss = criterion(logits, mask)
                else:
                    logits = model(rgb, thermal)
                    loss = criterion(logits, mask)

                running_val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_metrics.update(preds, mask)

                # Save a validation sample visualization periodically
                if not sample_visual_saved and (epoch % args.save_visuals_every == 0 or epoch == 1):
                    sample_rgb = denormalize_rgb(rgb[0].cpu().numpy(), cfg.dataset.rgb_mean, cfg.dataset.rgb_std)
                    sample_th = denormalize_thermal(thermal[0].cpu().numpy(), cfg.dataset.thermal_mean, cfg.dataset.thermal_std)
                    sample_gt = mask[0].cpu().numpy()
                    sample_pred = preds[0].cpu().numpy()
                    sample_name = batch["name"][0]

                    vis_path = os.path.join(visuals_dir, f"epoch_{epoch:03d}_{sample_name}.png")
                    create_comparison_plot(
                        sample_rgb, sample_th, sample_gt, sample_pred,
                        save_path=vis_path,
                        title=f"Epoch {epoch} Validation - Sample {sample_name}"
                    )
                    sample_visual_saved = True

        val_loss = running_val_loss / len(val_loader)
        scores = val_metrics.compute()
        val_acc = scores["pixel_accuracy"]
        val_mIoU = scores["mIoU"]
        val_mIoU_fg = scores["mIoU_fg"]

        epoch_duration = time.time() - epoch_start

        # Record History
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pixel_acc"].append(val_acc)
        history["val_mIoU"].append(val_mIoU)
        history["val_mIoU_fg"].append(val_mIoU_fg)
        history["lr"].append(current_lr)

        # Append to CSV
        with open(csv_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{val_mIoU:.4f}", f"{val_mIoU_fg:.4f}", f"{current_lr:.6f}"])

        # Write to JSON
        with open(json_log_path, "w") as f:
            json.dump(history, f, indent=2)

        # Update training curve plots after every epoch
        plot_training_history(history, save_dir=args.save_dir)

        # Checkpoint saving & Early stopping
        is_best = val_mIoU > best_mIoU
        if is_best:
            best_mIoU = val_mIoU

        ckpt_mgr.save(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            metrics={"mIoU": val_mIoU, "mIoU_fg": val_mIoU_fg, "pixel_acc": val_acc, "val_loss": val_loss},
            is_best=is_best
        )

        # Print Epoch Summary
        best_marker = " [★ NEW BEST mIoU]" if is_best else ""
        print(f"Epoch {epoch:03d}/{args.epochs:03d} ({epoch_duration:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"PA: {val_acc*100:.2f}% | mIoU: {val_mIoU*100:.2f}% (FG: {val_mIoU_fg*100:.2f}%){best_marker}")
        sys.stdout.flush()

        # Check early stopping
        improved = early_stopping.step(val_mIoU)
        if early_stopping.early_stop:
            print(f"\n[Early Stopping Triggered] No improvement in mIoU for {args.patience} consecutive epochs.")
            break

    total_time = time.time() - start_time
    print("\n" + "=" * 65)
    print("                    TRAINING COMPLETE")
    print("=" * 65)
    print(f"Total Time Elapsed   : {total_time / 60:.2f} minutes")
    print(f"Best Validation mIoU : {best_mIoU * 100:.2f}%")
    print(f"Best Checkpoint      : {os.path.join(args.save_dir, 'best_model.pth')}")
    print(f"Training Curves Plot : {os.path.join(args.save_dir, 'training_dashboard.png')}")
    print(f"Metrics Log (CSV)    : {csv_log_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
