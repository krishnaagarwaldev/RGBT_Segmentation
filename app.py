"""
CSFNet-2: Real-Time Multispectral (RGBT) Semantic Segmentation Web App
Supports both Single-Image and Live Video Stream Inference.
Powered exclusively by CSFNet-2 (STDC2 Backbone, ImageNet Pretrained, 19.36M Params).
"""

import os
import io
import time
import tempfile
import cv2
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import streamlit as st

from configs.default_config import get_default_config
from models.csfnet import build_csfnet
from utils.visualization import colorize_mask, CLASS_COLORS, CLASS_NAMES

# ------------------------------------------------------------------------------
# Page Configuration & Aesthetics
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="CSFNet-2 RGBT Segmentation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #50E3C2, #4A90E2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #8E9AAF;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        background: #181920;
        border: 1px solid #2B2D3A;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #50E3C2;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #8E9AAF;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .class-badge {
        display: inline-block;
        padding: 5px 10px;
        margin: 3px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Model Initialization (Cached)
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading CSFNet-2 weights...")
def get_model(device_str: str):
    """Instantiate and load CSFNet-2 best checkpoint."""
    device = torch.device(device_str)
    model = build_csfnet(
        backbone="STDCNet1446",
        pretrained=False,
        num_classes=9,
        dataset="MFNet"
    ).to(device)

    candidates = [
        "runs/csfnet2_pretrained/best_model.pth",
        "results/csfnet2_best_model.pth"
    ]
    ckpt_path = next((p for p in candidates if os.path.exists(p)), None)

    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        status = f"Loaded `{ckpt_path}`"
    else:
        status = "No checkpoint found. Running with initialized weights."

    model.eval()
    return model, status


def apply_thermal_colormap(gray_arr: np.ndarray, colormap_name: str = "inferno") -> np.ndarray:
    """Transform 1-channel thermal radiation to RGB pseudo-color."""
    norm = np.clip(gray_arr.astype(np.float32) / 255.0, 0.0, 1.0)
    try:
        import matplotlib as mpl
        cmap = mpl.colormaps[colormap_name]
    except Exception:
        try:
            cmap = cm.get_cmap(colormap_name)
        except Exception:
            return np.stack([gray_arr] * 3, axis=-1)
    colored = cmap(norm)[:, :, :3]
    return (colored * 255).astype(np.uint8)


# ------------------------------------------------------------------------------
# Sidebar Settings
# ------------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Model Settings")
st.sidebar.markdown("**Active Architecture:**")
st.sidebar.info("🚀 **CSFNet-2** (STDC2 Backbone)\n* 19.36M Parameters\n* 49.39% Test mIoU\n* ~72 FPS Real-Time")

device_choice = "cuda" if torch.cuda.is_available() else "cpu"
device = st.sidebar.selectbox("Inference Device:", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"], index=0)
if device == "cuda":
    st.sidebar.caption(f"⚡ GPU: {torch.cuda.get_device_name(0)}")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎨 Visual Controls")
overlay_alpha = st.sidebar.slider("Mask Overlay Opacity:", min_value=0.1, max_value=0.9, value=0.55, step=0.05)
thermal_palette = st.sidebar.selectbox("Thermal Palette:", ["inferno", "turbo", "plasma", "magma", "viridis", "gray"], index=0)

cfg = get_default_config()
model, load_msg = get_model(device)
st.sidebar.caption(f"Status: {load_msg}")


# ------------------------------------------------------------------------------
# Helper for Single Frame Inference
# ------------------------------------------------------------------------------
def predict_single_frame(rgb_np: np.ndarray, thermal_np: np.ndarray):
    """
    Given rgb_np (H, W, 3) in RGB and thermal_np (H, W) in grayscale,
    returns (pred_class_mask, colored_mask, blended_overlay, latency_ms)
    """
    pil_rgb = Image.fromarray(rgb_np, mode="RGB")
    pil_th = Image.fromarray(thermal_np, mode="L")

    t_rgb = TF.to_tensor(pil_rgb)
    t_rgb = TF.normalize(t_rgb, mean=list(cfg.dataset.rgb_mean), std=list(cfg.dataset.rgb_std)).unsqueeze(0).to(device)

    t_th = TF.to_tensor(pil_th)
    t_th = TF.normalize(t_th, mean=list(cfg.dataset.thermal_mean), std=list(cfg.dataset.thermal_std)).unsqueeze(0).to(device)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        output_logits = model(t_rgb, t_th)
        pred_mask = torch.argmax(output_logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    if device == "cuda":
        torch.cuda.synchronize()
    lat_ms = (time.perf_counter() - t0) * 1000.0

    colored_mask = colorize_mask(pred_mask)
    blended = (rgb_np * (1.0 - overlay_alpha) + colored_mask * overlay_alpha).astype(np.uint8)
    return pred_mask, colored_mask, blended, lat_ms


# ------------------------------------------------------------------------------
# Main UI Layout
# ------------------------------------------------------------------------------
st.markdown('<div class="main-header">Multimodal RGBT Semantic Segmentation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time scene parsing using paired Visible RGB + Thermal Infrared imaging with CSFNet-2.</div>', unsafe_allow_html=True)

tab_image, tab_video, tab_benchmark = st.tabs([
    "🖼️ Image Segmentation",
    "🎬 Live Video Stream",
    "📊 Model Performance"
])

# ==============================================================================
# TAB 1: Image Segmentation
# ==============================================================================
with tab_image:
    st.markdown("### 1. Upload Multispectral Image")
    
    up_mode = st.radio(
        "Upload Format:",
        ["Single 4-Channel RGBT Image (.png)", "Two Separate Files (RGB Color + Thermal Grayscale)"],
        horizontal=True,
        key="image_up_mode"
    )

    rgb_array, thermal_array = None, None
    sample_name = "Uploaded_Scene"

    if up_mode == "Single 4-Channel RGBT Image (.png)":
        st.caption("Upload a single `.png` where channels 0, 1, 2 are Visible RGB and channel 3 is Thermal infrared.")
        f_single = st.file_uploader("Upload 4-Channel RGBT Image (.png)", type=["png"], key="single_4ch_uploader")
        if f_single:
            img_4ch = Image.open(f_single)
            if img_4ch.mode != "RGBA":
                st.error("Uploaded image must have 4 channels (RGBA format).")
            else:
                arr_4ch = np.array(img_4ch.resize((640, 480)))
                rgb_array = arr_4ch[:, :, :3]
                thermal_array = arr_4ch[:, :, 3]
                sample_name = os.path.splitext(f_single.name)[0]

    else:
        st.caption("Upload separate RGB and Thermal image files.")
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            f_rgb = st.file_uploader("1. Visible Camera (RGB Color Image)", type=["png", "jpg", "jpeg"], key="img_rgb_up")
        with col_up2:
            f_th = st.file_uploader("2. Thermal Camera (Grayscale / LWIR Image)", type=["png", "jpg", "jpeg"], key="img_th_up")

        if f_rgb and f_th:
            rgb_img = Image.open(f_rgb).convert("RGB").resize((640, 480))
            th_img = Image.open(f_th).convert("L").resize((640, 480))
            rgb_array = np.array(rgb_img)
            thermal_array = np.array(th_img)
            sample_name = os.path.splitext(f_rgb.name)[0]

    # Results Section
    if rgb_array is not None and thermal_array is not None:
        st.markdown("---")
        st.markdown("### 2. Segmentation Results")

        pred_class_mask, colored_mask, blended_overlay, duration_ms = predict_single_frame(rgb_array, thermal_array)
        fps_val = 1000.0 / duration_ms if duration_ms > 0 else 0
        colored_thermal = apply_thermal_colormap(thermal_array, thermal_palette)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{duration_ms:.1f} ms</div><div class="stat-label">Latency</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{fps_val:.1f} FPS</div><div class="stat-label">Inference Speed</div></div>', unsafe_allow_html=True)
        with s3:
            st.markdown(f'<div class="stat-box"><div class="stat-number">480 × 640</div><div class="stat-label">Resolution</div></div>', unsafe_allow_html=True)
        with s4:
            detected_classes = [CLASS_NAMES[c] for c in np.unique(pred_class_mask) if c != 0]
            st.markdown(f'<div class="stat-box"><div class="stat-number">{len(detected_classes)}</div><div class="stat-label">Objects Found</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.image(blended_overlay, caption=f"✨ Blended Overlay (Alpha = {overlay_alpha:.2f})", use_container_width=True)
            st.image(rgb_array, caption="📷 Visible RGB Input", use_container_width=True)

        with col_res2:
            st.image(colored_mask, caption="🎯 Predicted Semantic Mask (9 Classes)", use_container_width=True)
            st.image(colored_thermal, caption=f"🌡️ Thermal Infrared ({thermal_palette.capitalize()} Colormap)", use_container_width=True)

        st.markdown("### 3. Export Predictions")
        d_col1, d_col2 = st.columns(2)

        buf_blend = io.BytesIO()
        Image.fromarray(blended_overlay).save(buf_blend, format="PNG")
        with d_col1:
            st.download_button(
                label="⬇️ Download Overlay Image (PNG)",
                data=buf_blend.getvalue(),
                file_name=f"{sample_name}_csfnet2_overlay.png",
                mime="image/png"
            )

        buf_mask = io.BytesIO()
        Image.fromarray(colored_mask).save(buf_mask, format="PNG")
        with d_col2:
            st.download_button(
                label="⬇️ Download Semantic Mask (PNG)",
                data=buf_mask.getvalue(),
                file_name=f"{sample_name}_csfnet2_mask.png",
                mime="image/png"
            )

        st.markdown("---")
        st.markdown("### 🏷️ Detected Classes & Distribution")
        total_px = pred_class_mask.size
        leg_cols = st.columns(len(CLASS_NAMES))
        for i, c_name in enumerate(CLASS_NAMES):
            c_rgb = CLASS_COLORS[i]
            hex_c = f"#{c_rgb[0]:02x}{c_rgb[1]:02x}{c_rgb[2]:02x}"
            px_count = np.sum(pred_class_mask == i)
            pct = (px_count / total_px) * 100
            is_present = px_count > 0

            with leg_cols[i]:
                opacity_style = "opacity: 1.0;" if is_present else "opacity: 0.35;"
                st.markdown(
                    f"<div style='background-color: {hex_c}; {opacity_style} text-align: center; color: white; padding: 6px; border-radius: 6px; font-weight: bold; font-size: 11px;'>"
                    f"{c_name}</div>"
                    f"<div style='text-align: center; font-size: 12px; margin-top: 4px; color: {'#50E3C2' if is_present else '#777'};'>{pct:.1f}%</div>",
                    unsafe_allow_html=True
                )


# ==============================================================================
# TAB 2: Live Video Stream Inference
# ==============================================================================
with tab_video:
    st.markdown("### 🎬 Real-Time Video Segmentation & Live Stream")
    st.caption("Process driving videos frame-by-frame with CSFNet-2 and visualize real-time multispectral overlays.")

    # 1. Video Upload
    st.markdown("### 1. Upload Video Input")
    up_v_mode = st.radio(
        "Upload Format:",
        [
            "Side-by-Side Video (.mp4) — Left half RGB, Right half Thermal",
            "Two Separate Videos (.mp4) — RGB Video + Thermal Video",
            "Single Standard RGB Video (.mp4) — Uses intensity fallback for thermal"
        ]
    )

    cap_rgb = None
    cap_th = None
    is_side_by_side = False
    is_single_fallback = False
    temp_files_to_clean = []
    video_title = "RGBT_Stream"

    if up_v_mode.startswith("Side-by-Side"):
        st.caption("Upload a single `.mp4` video where the left half is RGB color and the right half is Thermal infrared.")
        v_file = st.file_uploader("Upload Side-by-Side Video (.mp4)", type=["mp4", "avi", "mov"], key="v_sbs_up")
        if v_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(v_file.read())
            tfile.close()
            temp_files_to_clean.append(tfile.name)
            cap_rgb = cv2.VideoCapture(tfile.name)
            is_side_by_side = True
            video_title = os.path.splitext(v_file.name)[0]

    elif up_v_mode.startswith("Two Separate"):
        st.caption("Upload two synchronized video files: one for visible RGB and one for thermal infrared.")
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            v_rgb_file = st.file_uploader("1. Visible RGB Video (.mp4)", type=["mp4", "avi", "mov"], key="v_rgb_up")
        with col_v2:
            v_th_file = st.file_uploader("2. Thermal Infrared Video (.mp4)", type=["mp4", "avi", "mov"], key="v_th_up")

        if v_rgb_file and v_th_file:
            t_rgb = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t_rgb.write(v_rgb_file.read())
            t_rgb.close()
            temp_files_to_clean.append(t_rgb.name)
            cap_rgb = cv2.VideoCapture(t_rgb.name)

            t_th = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t_th.write(v_th_file.read())
            t_th.close()
            temp_files_to_clean.append(t_th.name)
            cap_th = cv2.VideoCapture(t_th.name)
            video_title = os.path.splitext(v_rgb_file.name)[0]

    else:
        st.caption("Upload a standard single-stream RGB video (e.g. dashcam or phone video). Thermal channel will be approximated from intensity.")
        v_single_file = st.file_uploader("Upload Regular RGB Video (.mp4)", type=["mp4", "avi", "mov"], key="v_single_up")
        if v_single_file:
            t_single = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            t_single.write(v_single_file.read())
            t_single.close()
            temp_files_to_clean.append(t_single.name)
            cap_rgb = cv2.VideoCapture(t_single.name)
            is_single_fallback = True
            video_title = os.path.splitext(v_single_file.name)[0]

    # Stream Controls
    if cap_rgb is not None and cap_rgb.isOpened():
        st.markdown("---")
        st.markdown("### Stream Settings & Live Display")

        c_opt1, c_opt2, c_opt3, c_opt4 = st.columns(4)
        with c_opt1:
            view_layout = st.selectbox(
                "Display Viewport Layout:",
                [
                    "Side-by-Side: Original RGB vs Segmented Overlay",
                    "Triple View: RGB | Thermal | Segmented Overlay",
                    "Segmented Overlay Only"
                ]
            )
        with c_opt2:
            playback_fps = st.slider(
                "Stream Speed (FPS):",
                min_value=1,
                max_value=25,
                value=5,
                help="1-4 FPS: Slow-motion detailed inspection. 5-10 FPS: Natural driving speed. 15-25 FPS: Fast stream."
            )
        with c_opt3:
            frame_stride = st.selectbox("Frame Stride:", [1, 2, 3], index=0,
                                        help="1: Process all frames. 2: Process every 2nd frame.")
        with c_opt4:
            max_frames = st.slider("Max Frames to Stream:", min_value=10, max_value=150, value=50, step=5)

        start_stream = st.button("▶️ Start Live Stream", type="primary", key="btn_start_stream")

        if start_stream:
            st.markdown("#### 🔴 Live Streaming Inference")

            # HUD Stat Placeholders
            h1, h2, h3, h4 = st.columns(4)
            box_fps = h1.empty()
            box_lat = h2.empty()
            box_frame = h3.empty()
            box_cls = h4.empty()

            progress_bar = st.progress(0.0)
            viewport = st.empty()
            classes_strip = st.empty()

            # Output video writer setup
            out_path = os.path.join(tempfile.gettempdir(), f"{video_title}_segmented_output.mp4")
            out_writer = None

            frame_idx = 0
            processed_count = 0
            fps_history = []

            try:
                while cap_rgb.isOpened() and processed_count < max_frames:
                    t_frame_start = time.perf_counter()
                    ret_rgb, raw_frame_rgb = cap_rgb.read()
                    if not ret_rgb:
                        break

                    # Retrieve thermal frame
                    if is_side_by_side:
                        # Left half = RGB, Right half = Thermal
                        h_f, w_f, _ = raw_frame_rgb.shape
                        mid_w = w_f // 2
                        frame_bgr = raw_frame_rgb[:, :mid_w]
                        raw_th = raw_frame_rgb[:, mid_w:]
                        frame_th_gray = cv2.cvtColor(raw_th, cv2.COLOR_BGR2GRAY)
                    elif cap_th is not None:
                        ret_th, raw_frame_th = cap_th.read()
                        if not ret_th:
                            break
                        frame_bgr = raw_frame_rgb
                        frame_th_gray = cv2.cvtColor(raw_frame_th, cv2.COLOR_BGR2GRAY)
                    elif is_single_fallback:
                        frame_bgr = raw_frame_rgb
                        frame_th_gray = cv2.cvtColor(raw_frame_rgb, cv2.COLOR_BGR2GRAY)
                    else:
                        break

                    frame_idx += 1
                    if frame_idx % frame_stride != 0:
                        continue

                    # Resize to model resolution
                    frame_bgr = cv2.resize(frame_bgr, (640, 480))
                    frame_th_gray = cv2.resize(frame_th_gray, (640, 480))
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR if False else cv2.COLOR_BGR2RGB)

                    # Model prediction
                    pred_mask, col_mask, overlay, lat_ms = predict_single_frame(frame_rgb, frame_th_gray)
                    instant_fps = 1000.0 / lat_ms if lat_ms > 0 else 0
                    fps_history.append(instant_fps)
                    avg_fps = np.mean(fps_history[-10:])
                    processed_count += 1

                    # Compose output frame according to layout
                    col_th_viz = apply_thermal_colormap(frame_th_gray, thermal_palette)

                    if view_layout.startswith("Side-by-Side"):
                        display_img = np.hstack([frame_rgb, overlay])
                    elif view_layout.startswith("Triple"):
                        display_img = np.hstack([frame_rgb, col_th_viz, overlay])
                    else:
                        display_img = overlay

                    # Initialize video writer once dimensions are known
                    if out_writer is None:
                        out_h, out_w, _ = display_img.shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out_writer = cv2.VideoWriter(out_path, fourcc, playback_fps, (out_w, out_h))

                    # Write frame to video
                    out_writer.write(cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))

                    # Update live viewport & HUD
                    viewport.image(display_img, use_container_width=True)
                    progress_bar.progress(min(processed_count / max_frames, 1.0))

                    box_fps.markdown(f'<div class="stat-box"><div class="stat-number">{playback_fps} FPS</div><div class="stat-label">Playback Pacing</div></div>', unsafe_allow_html=True)
                    box_lat.markdown(f'<div class="stat-box"><div class="stat-number">{lat_ms:.1f} ms</div><div class="stat-label">Model Latency</div></div>', unsafe_allow_html=True)
                    box_frame.markdown(f'<div class="stat-box"><div class="stat-number">{processed_count}/{max_frames}</div><div class="stat-label">Processed</div></div>', unsafe_allow_html=True)

                    active_cls = [CLASS_NAMES[c] for c in np.unique(pred_mask) if c != 0]
                    box_cls.markdown(f'<div class="stat-box"><div class="stat-number">{len(active_cls)}</div><div class="stat-label">Active Objects</div></div>', unsafe_allow_html=True)

                    # Dynamic class tags
                    badge_html = "".join([
                        f"<span style='background: #252836; border: 1px solid #50E3C2; color: #50E3C2; padding: 4px 10px; border-radius: 6px; margin: 3px; font-size: 11px; font-weight: bold;'>{name}</span>"
                        for name in active_cls
                    ])
                    disp_badges = badge_html if badge_html else '<span style="color: #777;">None</span>'
                    classes_strip.markdown(f"<div style='margin-top: 6px;'><b>Detected in Current Frame:</b> {disp_badges}</div>", unsafe_allow_html=True)

                    # Regulate stream speed to match selected FPS
                    if playback_fps > 0:
                        target_dt = 1.0 / playback_fps
                        elapsed_dt = time.perf_counter() - t_frame_start
                        if elapsed_dt < target_dt:
                            time.sleep(target_dt - elapsed_dt)

                st.success(f"🎉 Live stream finished! Processed {processed_count} frames at {playback_fps} FPS.")

            finally:
                if out_writer is not None:
                    out_writer.release()
                if cap_rgb is not None:
                    cap_rgb.release()
                if cap_th is not None:
                    cap_th.release()

            # Video Download & Playback
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                st.markdown("---")
                st.markdown("### 📥 Download Segmented Video")
                with open(out_path, "rb") as vf:
                    v_bytes = vf.read()

                st.download_button(
                    label="⬇️ Download Segmented Video (.mp4)",
                    data=v_bytes,
                    file_name=f"{video_title}_csfnet2_segmented.mp4",
                    mime="video/mp4"
                )

        # Cleanup temporary files
        for f_path in temp_files_to_clean:
            try:
                if os.path.exists(f_path):
                    os.remove(f_path)
            except Exception:
                pass


# ==============================================================================
# TAB 3: Benchmark Details
# ==============================================================================
with tab_benchmark:
    st.markdown("### 🏆 CSFNet-2 Benchmark Performance")
    st.markdown("Empirical test results on the **MFNet** dataset using our trained model on an **NVIDIA GeForce RTX 3050 Laptop GPU**:")

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown("""
        #### Overall Performance
        * **Overall Test mIoU:** **49.39%** *(+10.93% over baseline)*
        * **Foreground mIoU:** **43.41%** *(+12.12% over baseline)*
        * **Pixel Accuracy:** **97.16%**
        * **Real-Time Speed:** **~72.7 FPS** (13.7 ms latency)
        * **Parameters:** **19.36M**
        """)

    with t_col2:
        st.markdown("""
        #### Day vs Night Sensitivity
        * ☀️ **Daytime mIoU (`test_day`):** **49.79%** (Car IoU: 83.51%)
        * 🌙 **Nighttime mIoU (`test_night`):** **47.64%** (Person IoU: **70.56%**)
        
        > **Day/Night Gap:** Only **2.15%**, demonstrating robust illumination invariance. Under zero-light conditions, thermal infrared radiation ensures pedestrians remain clearly detectable.
        """)

    st.markdown("---")
    st.markdown("#### Per-Class Foreground IoU on MFNet Benchmark")
    st.markdown("""
    | Class | Color Badge | IoU (%) | Baseline Comparison |
    | :--- | :---: | :---: | :---: |
    | **Car** | 🟦 Blue | **86.13%** | +2.85% |
    | **Person / Pedestrian** | 🟨 Yellow | **67.42%** | +6.29% |
    | **Bike / Bicycle** | 🟩 Green | **60.33%** | +7.80% |
    | **Curve** | 🟧 Orange | **44.97%** | +15.22% |
    | **Car Stop** | 🟪 Purple | **27.81%** | +12.40% |
    | **Color Cone** | 🟥 Red | **36.91%** | +19.31% |
    | **Bump** | 🟫 Brown | **42.25%** | +14.65% |
    | **Guardrail** | ⬜ Silver | **8.52%** | +7.92% |
    """)
