"""
Computer Vision Pipeline using OpenCV
Handles image preprocessing, metadata extraction, edge/contour detection,
contrast enhancement (CLAHE), and color histogram computation.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Safely decode raw image bytes into an OpenCV BGR numpy array,
    handling EXIF orientation tags automatically.
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img = ImageOps.exif_transpose(pil_img)
        # Convert PIL (RGB / RGBA / L) to OpenCV BGR
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        rgb_array = np.array(pil_img)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array
    except Exception as e:
        # Fallback to direct cv2.imdecode
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image bytes: {e}")
        return img


def get_image_metadata(cv_img: np.ndarray, file_size_bytes: int = 0) -> dict:
    """Extract resolution, aspect ratio, channels, and brightness metrics."""
    h, w = cv_img.shape[:2]
    c = cv_img.shape[2] if len(cv_img.shape) > 2 else 1
    
    # Calculate average brightness and contrast
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if c > 1 else cv_img
    mean_brightness = float(np.mean(gray))
    std_contrast = float(np.std(gray))
    
    aspect_ratio = round(w / h, 2) if h > 0 else 1.0
    
    # Size formatting
    if file_size_bytes >= 1024 * 1024:
        size_str = f"{file_size_bytes / (1024 * 1024):.2f} MB"
    elif file_size_bytes >= 1024:
        size_str = f"{file_size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{file_size_bytes} B"

    return {
        "width": w,
        "height": h,
        "channels": c,
        "aspect_ratio": aspect_ratio,
        "file_size": size_str,
        "brightness": round(mean_brightness, 1),
        "contrast": round(std_contrast, 1),
    }


def apply_enhancements(cv_img: np.ndarray, clahe: bool = False, sharpen: bool = False, denoise: bool = False) -> np.ndarray:
    """Apply optional computer vision filters."""
    result = cv_img.copy()

    if denoise:
        # Fast bilateral filter preserving edges
        result = cv2.bilateralFilter(result, d=7, sigmaColor=50, sigmaSpace=50)

    if clahe:
        # Convert to LAB, apply CLAHE on L-channel, convert back
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe_op = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_clahe = clahe_op.apply(l_chan)
        lab_clahe = cv2.merge((l_clahe, a_chan, b_chan))
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    if sharpen:
        # Unsharp masking kernel
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        result = cv2.filter2D(result, -1, kernel)

    return result


def detect_edges_canny(cv_img: np.ndarray, low: int = 50, high: int = 150) -> str:
    """Generate Canny edge visualization rendered with cyan aesthetic glowing edges."""
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blurred, low, high)

    # Colorize edges (neon cyan on dark slate background)
    h, w = edges.shape
    colored_edges = np.zeros((h, w, 3), dtype=np.uint8)
    colored_edges[edges > 0] = [238, 175, 43]  # Neon Cyan in BGR

    return encode_image_base64(colored_edges, fmt="png")


def detect_contours_and_boxes(cv_img: np.ndarray, max_contours: int = 6) -> Tuple[str, List[Dict[str, Any]]]:
    """Detect salient regions/contours and draw futuristic bounding boxes with labels."""
    h, w = cv_img.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to find prominent regions
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 3
    )

    find_res = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = find_res[0] if len(find_res) == 2 else find_res[1]
    
    # Filter and sort by area
    min_area = (h * w) * 0.015  # At least 1.5% of total image
    filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    filtered = sorted(filtered, key=cv2.contourArea, reverse=True)[:max_contours]

    annotated = cv_img.copy()
    detections = []

    # Futuristic colors in BGR
    palette = [
        (255, 105, 180),  # Hot pink
        (0, 220, 255),    # Vibrant yellow-gold
        (240, 160, 40),   # Electric cyan
        (50, 205, 50),    # Lime green
        (200, 100, 255),  # Violet
        (255, 191, 0)     # Deep sky blue
    ]

    for idx, cnt in enumerate(filtered):
        color = palette[idx % len(palette)]
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        detections.append({
            "id": idx + 1,
            "x": int(x),
            "y": int(y),
            "width": int(bw),
            "height": int(bh),
            "area_percentage": round((area / (h * w)) * 100, 1)
        })

        # Draw smooth contour outline
        cv2.drawContours(annotated, [cnt], -1, color, 2)
        
        # Draw sleek corner brackets for a HUD feel
        corner_len = min(bw, bh) // 4
        # Top-left
        cv2.line(annotated, (x, y), (x + corner_len, y), color, 3)
        cv2.line(annotated, (x, y), (x, y + corner_len), color, 3)
        # Top-right
        cv2.line(annotated, (x + bw, y), (x + bw - corner_len, y), color, 3)
        cv2.line(annotated, (x + bw, y), (x + bw, y + corner_len), color, 3)
        # Bottom-left
        cv2.line(annotated, (x, y + bh), (x + corner_len, y + bh), color, 3)
        cv2.line(annotated, (x, y + bh), (x, y + bh - corner_len), color, 3)
        # Bottom-right
        cv2.line(annotated, (x + bw, y + bh), (x + bw - corner_len, y + bh), color, 3)
        cv2.line(annotated, (x + bw, y + bh), (x + bw, y + bh - corner_len), color, 3)

        # Badge label
        label = f"ROI #{idx + 1}"
        cv2.rectangle(annotated, (x, max(0, y - 22)), (x + 75, max(0, y)), color, -1)
        cv2.putText(
            annotated, label, (x + 5, max(15, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
        )

    return encode_image_base64(annotated, fmt="jpeg"), detections


def compute_histogram_visualization(cv_img: np.ndarray) -> Tuple[str, Dict[str, Any]]:
    """
    Compute RGB color histograms and render a stylish dark-themed visualization.
    Also returns normalized data series for frontend charts.
    """
    hist_w, hist_h = 512, 160
    hist_canvas = np.full((hist_h, hist_w, 3), 20, dtype=np.uint8)  # Dark charcoal bg

    colors = [
        ('Blue', (255, 120, 50), 0),   # B
        ('Green', (50, 220, 100), 1),  # G
        ('Red', (50, 70, 255), 2)      # R
    ]
    
    chart_data = {'blue': [], 'green': [], 'red': []}
    bin_count = 256

    for name, bgr_color, chan in colors:
        hist = cv2.calcHist([cv_img], [chan], None, [bin_count], [0, 256])
        cv2.normalize(hist, hist, 0, hist_h - 20, cv2.NORM_MINMAX)
        hist_flat = hist.flatten()
        
        # Store for JSON response (sampled down to 64 points for compact payload)
        sampled = [round(float(hist_flat[i * 4]), 2) for i in range(64)]
        chart_data[name.lower()] = sampled

        # Draw smooth line graph on canvas
        for i in range(1, bin_count):
            pt1 = (int((i - 1) * (hist_w / bin_count)), hist_h - int(hist_flat[i - 1]) - 10)
            pt2 = (int(i * (hist_w / bin_count)), hist_h - int(hist_flat[i]) - 10)
            cv2.line(hist_canvas, pt1, pt2, bgr_color, 2, cv2.LINE_AA)

    # Add subtle grid lines
    for y in [40, 80, 120]:
        cv2.line(hist_canvas, (0, y), (hist_w, y), (40, 40, 45), 1)

    return encode_image_base64(hist_canvas, fmt="png"), chart_data


def encode_image_base64(cv_img: np.ndarray, fmt: str = "jpeg", quality: int = 88) -> str:
    """Encode OpenCV image matrix into a Base64 data URL string."""
    if fmt.lower() in ("jpg", "jpeg"):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", cv_img, encode_param)
        mime = "image/jpeg"
    else:
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 4]
        success, buffer = cv2.imencode(".png", cv_img, encode_param)
        mime = "image/png"

    if not success:
        raise ValueError("Failed to encode OpenCV image to buffer")

    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:{mime};base64,{b64_str}"
