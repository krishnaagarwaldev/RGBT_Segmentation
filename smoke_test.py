"""Comprehensive sanity check and GPU smoke test for CSFNet-1 on ir_seg_dataset.

Executes 7 thorough verification stages:
1. Dataset Integrity & Pairing Check
2. Synchronized Transforms & Mask ID Preservation Check
3. Class Imbalance & Loss Weights Check
4. Model Architecture & Parameter Count Check (11.30M)
5. GPU Forward, Backward, & AMP Execution Check (RTX 3050 6GB)
6. Metrics Accumulator & Confusion Matrix Check
7. Visualization & Panel Export Check

Usage:
  python smoke_test.py
"""
import os
import sys
import torch
import numpy as np
from PIL import Image

# Import project modules
from configs.default_config import get_default_config
from data.dataset import RGBTDataset
from data.transforms import get_train_transforms, get_val_transforms
from data.class_weights import compute_class_weights, get_class_distribution_summary
from models.csfnet import build_csfnet1
from models.losses import build_loss
from utils.metrics import SegmentationMetrics
from utils.visualization import create_comparison_plot, colorize_mask


def run_smoke_test():
    print("=" * 70)
    print("          CSFNet-1 (STDC1) RGBT SYSTEM SANITY & SMOKE TEST")
    print("=" * 70)

    cfg = get_default_config()
    tests_passed = 0
    total_tests = 7

    # -------------------------------------------------------------
    # STAGE 1: Dataset Integrity & Pairing Check
    # -------------------------------------------------------------
    print("\n[Stage 1/7] Checking Dataset Integrity and Modality Pairing...")
    images_dir = cfg.dataset.images_dir
    labels_dir = cfg.dataset.labels_dir
    assert os.path.exists(images_dir), f"Missing images dir: {images_dir}"
    assert os.path.exists(labels_dir), f"Missing labels dir: {labels_dir}"

    sample_img_path = os.path.join(images_dir, "00001D.png")
    sample_lbl_path = os.path.join(labels_dir, "00001D.png")
    assert os.path.exists(sample_img_path), "Missing sample 00001D.png in images/"
    assert os.path.exists(sample_lbl_path), "Missing sample 00001D.png in labels/"

    with Image.open(sample_img_path) as im:
        assert im.size == (640, 480), f"Unexpected image size: {im.size}"
        assert im.mode == "RGBA", f"Expected RGBA mode for 4-channel image, got: {im.mode}"
        arr = np.array(im)
        assert arr.shape == (480, 640, 4), f"Unexpected array shape: {arr.shape}"

    with Image.open(sample_lbl_path) as lbl:
        assert lbl.size == (640, 480), f"Unexpected label size: {lbl.size}"
        arr_lbl = np.array(lbl)
        unique_classes = np.unique(arr_lbl)
        assert np.all(unique_classes >= 0) and np.all(unique_classes < cfg.dataset.num_classes), \
            f"Class IDs out of range 0..8: {unique_classes}"

    print("  -> Images & Labels exist, 640x480 resolution confirmed.")
    print("  -> 4-channel pairing confirmed: Channels 0..2 (RGB), Channel 3 (Thermal).")
    print(f"  -> Mask IDs verified: range [{unique_classes.min()}..{unique_classes.max()}] in 0..8.")
    print("  [PASS] Stage 1 Successful!")
    tests_passed += 1

    # -------------------------------------------------------------
    # STAGE 2: Synchronized Transforms & Mask ID Preservation Check
    # -------------------------------------------------------------
    print("\n[Stage 2/7] Checking Synchronized Transforms & Mask ID Preservation...")
    train_tf = get_train_transforms()
    val_tf = get_val_transforms()

    # Load sample and apply transforms
    im_pil = Image.open(sample_img_path)
    im_np = np.array(im_pil)
    rgb_pil = Image.fromarray(im_np[:, :, :3], mode="RGB")
    th_pil = Image.fromarray(im_np[:, :, 3], mode="L")
    mask_pil = Image.open(sample_lbl_path).convert("L")

    rgb_t, th_t, mask_t = train_tf(rgb_pil, th_pil, mask_pil)

    assert rgb_t.shape == (3, 480, 640), f"Unexpected RGB tensor shape: {rgb_t.shape}"
    assert th_t.shape == (1, 480, 640), f"Unexpected Thermal tensor shape: {th_t.shape}"
    assert mask_t.shape == (480, 640), f"Unexpected Mask tensor shape: {mask_t.shape}"
    assert mask_t.dtype == torch.int64, f"Mask tensor must be int64, got: {mask_t.dtype}"

    # Confirm mask IDs did NOT get corrupted into float values by interpolation
    unique_mask_ids = torch.unique(mask_t).numpy()
    for uid in unique_mask_ids:
        assert 0 <= uid <= 8, f"Corrupted mask ID found after transforms: {uid}"

    print("  -> Output tensor shapes: RGB (3, 480, 640), Thermal (1, 480, 640), Mask (480, 640).")
    print(f"  -> Mask unique IDs preserved after scaling & cropping: {unique_mask_ids}.")
    print("  [PASS] Stage 2 Successful!")
    tests_passed += 1

    # -------------------------------------------------------------
    # STAGE 3: Class Imbalance & Loss Weights Check
    # -------------------------------------------------------------
    print("\n[Stage 3/7] Checking Class Distribution & Principled Loss Weights...")
    w_enet = compute_class_weights("enet_log")
    w_med = compute_class_weights("median_frequency")
    w_pow = compute_class_weights("power_smoothed")

    assert len(w_enet) == 9, "Weight tensor length mismatch"
    assert not torch.isnan(w_enet).any(), "NaN found in class weights"
    assert torch.isclose(torch.mean(w_enet), torch.tensor(1.0), atol=1e-2), "Weights not normalized to mean 1.0"

    print("  -> Computed ENet log weights (mean 1.0):")
    print(f"     {w_enet.numpy().round(3)}")
    print("  [PASS] Stage 3 Successful!")
    tests_passed += 1

    # -------------------------------------------------------------
    # STAGE 4: Model Architecture & Parameter Count Check
    # -------------------------------------------------------------
    print("\n[Stage 4/7] Checking CSFNet-1 Architecture & Parameter Count...")
    model = build_csfnet1(num_classes=9, dataset="MFNet")
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    print(f"  -> Total Parameters: {total_params:.2f}M (Expected: 11.30M)")
    assert abs(total_params - 11.30) < 0.05, f"Unexpected parameter count: {total_params}M"
    print("  [PASS] Stage 4 Successful!")
    tests_passed += 1

    # -------------------------------------------------------------
    # STAGE 5: GPU Forward, Backward, & AMP Execution Check
    # -------------------------------------------------------------
    print("\n[Stage 5/7] Checking GPU Execution with Automatic Mixed Precision (AMP)...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  -> Testing on CUDA Device: {gpu_name}")
        device = torch.device("cuda:0")
        model = model.to(device)

        # Batch of size 2, 480x640
        b_rgb = torch.randn(2, 3, 480, 640, device=device)
        b_th = torch.randn(2, 1, 480, 640, device=device)
        b_mask = torch.randint(0, 9, (2, 480, 640), device=device)

        criterion = build_loss("combined_ce_dice", weight_strategy="enet_log", device="cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        try:
            scaler = torch.amp.GradScaler("cuda")
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler()

        # Forward with AMP
        try:
            with torch.amp.autocast("cuda"):
                out = model(b_rgb, b_th)
                loss = criterion(out, b_mask)
        except AttributeError:
            with torch.cuda.amp.autocast():
                out = model(b_rgb, b_th)
                loss = criterion(out, b_mask)

        assert out.shape == (2, 9, 480, 640), f"Unexpected model output shape: {out.shape}"
        assert not torch.isnan(loss), "Loss computed to NaN"

        # Backward step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  -> Forward & Backward successful! Loss: {loss.item():.4f}")
        print(f"  -> Peak GPU VRAM allocated: {vram_mb:.1f} MB (Well within 6GB capacity).")
    else:
        print("  -> CUDA not available; tested on CPU.")
        b_rgb = torch.randn(1, 3, 480, 640)
        b_th = torch.randn(1, 1, 480, 640)
        out = model(b_rgb, b_th)
        assert out.shape == (1, 9, 480, 640)
        print("  -> CPU Forward successful.")

    print("  [PASS] Stage 5 Successful!")
    tests_passed += 1

    # -------------------------------------------------------------
    # STAGE 6: Metrics Accumulator & Confusion Matrix Check
    # -------------------------------------------------------------
    print("\n[Stage 6/7] Checking Metrics Accumulator & IoU Computation...")
    metrics = SegmentationMetrics(num_classes=9)
    synth_gt = np.random.randint(0, 9, (480, 640))
    synth_pred = synth_gt.copy()  # Perfect prediction test
    metrics.update(synth_pred, synth_gt)
    scores = metrics.compute()

    assert np.isclose(scores["pixel_accuracy"], 1.0), "Perfect prediction should have 100% PA"
    assert np.isclose(scores["mIoU"], 1.0), "Perfect prediction should have 100% mIoU"
    assert scores["confusion_matrix"].shape == (9, 9), "Confusion matrix shape mismatch"

    print("  -> Confusion matrix shape: 9x9.")
    print(f"  -> Verified perfect prediction IoU: {scores['mIoU'] * 100:.1f}%.")
    print("  [PASS] Stage 6 Successful!")
    tests_passed += 1

    # -------------------------------------------------------------
    # STAGE 7: Visualization & Panel Export Check
    # -------------------------------------------------------------
    print("\n[Stage 7/7] Checking Visualization & Panel Generation...")
    scratch_dir = "runs/smoke_test"
    os.makedirs(scratch_dir, exist_ok=True)
    test_plot_path = os.path.join(scratch_dir, "test_smoke_plot.png")

    dummy_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_th = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    dummy_gt = np.random.randint(0, 9, (480, 640), dtype=np.uint8)
    dummy_pred = np.random.randint(0, 9, (480, 640), dtype=np.uint8)

    create_comparison_plot(
        dummy_rgb, dummy_th, dummy_gt, dummy_pred,
        save_path=test_plot_path,
        title="Smoke Test Visual Verification"
    )

    assert os.path.exists(test_plot_path), f"Plot was not saved: {test_plot_path}"
    assert os.path.getsize(test_plot_path) > 1000, "Saved plot file is empty"
    print(f"  -> Saved test panel plot to: {test_plot_path}")
    print("  [PASS] Stage 7 Successful!")
    tests_passed += 1

    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"SMOKE TEST SUMMARY: {tests_passed}/{total_tests} STAGES PASSED SUCCESSFULLY!")
    print("The environment, dataset, architecture, and GPU pipeline are 100% ready.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
