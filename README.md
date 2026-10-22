# VisionPulse AI - Image Recognition & Object Classification System

A production-grade, real-time image recognition web application developed using **Python, TensorFlow / Keras, OpenCV, and Flask**.

---

## 🌟 Key Features

1. **Dual Neural Architectures (ImageNet 1,000 Classes)**:
   - **MobileNetV2**: Ultra-fast (~15–30 ms) inference optimized for real-time edge performance.
   - **ResNet-50**: Deep residual 50-layer architecture for complex, high-accuracy feature extraction.
2. **OpenCV Computer Vision Diagnostic Suite**:
   - **Canny Edge Detection**: Neon cyan wireframes isolating high-frequency boundaries and object silhouettes.
   - **Contour & ROI Detection**: Futuristic bounding brackets highlighting salient regions and calculated area percentages.
   - **Color Histogram Analysis**: Intensity distribution charts across Red, Green, and Blue channels.
   - **CLAHE Equalization**: Adaptive contrast enhancement for backlit or low-light images.
   - **Bilateral Filtering & Unsharp Masking**: Denoising and edge sharpening before classification.
3. **Grad-CAM Neural Attention Heatmaps**:
   - Computes gradients of the predicted class with respect to the final convolutional feature maps (`out_relu` / `conv5_block3_out`).
   - Generates a colored attention heatmap overlaid directly on the image to visualize *why* the model made its prediction.
4. **Multiple Input Modalities**:
   - **Drag & Drop Upload**: PNG, JPG, WEBP, BMP up to 16MB.
   - **Live Webcam Capture**: Integrated HTML5 video feed and shutter for snapshot classification.
   - **One-Click Benchmark Gallery**: Instant testing with Golden Retriever, Sports Car, Monarch Butterfly, and Espresso samples.
5. **Modern Glassmorphic Dark-Mode UI**:
   - Responsive CSS layout, animated probability bars, and one-click report export to clipboard.
6. **Custom Dataset Training Pipeline**:
   - Includes `train_custom_model.py` for end-to-end transfer learning on your own dataset categories.

---

## 🚀 Quick Start Guide

### 1. Requirements & Prerequisites
Ensure you have Python 3.10+ installed with `pip`.

Install dependencies:
```bash
py -m pip install -r requirements.txt
```

### 2. Launching the Web Application
Run the Flask server:
```bash
py app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🧪 Training Your Own Custom Model

You can train a custom image classifier on your own folders using `train_custom_model.py`:

### Step 1: Generate a Sample Dataset (or use your own)
```bash
py train_custom_model.py --generate_samples
```
This generates synthetic sample categories (`circles`, `rectangles`, `triangles`) in `dataset_sample/`.

### Step 2: Run Transfer Learning & Fine-Tuning
```bash
py train_custom_model.py --data_dir dataset_sample/ --epochs 5 --batch_size 16
```
This freezes the base MobileNetV2 feature extractor, trains a new classification head, performs fine-tuning on the top convolutional layers, and saves the trained model to `custom_model.keras`.

---

## 📡 API Reference

### `POST /api/classify`
Classifies an image and returns Top-5 predictions plus OpenCV diagnostics.

**Supported Input Formats:**
- **Multipart Form Data**: `file` (image file), `model` (`mobilenet_v2` or `resnet50`), `clahe` (`true`/`false`), `sharpen` (`true`/`false`), `denoise` (`true`/`false`)
- **JSON Payload**:
  ```json
  {
    "sample_id": "golden_retriever",
    "model": "mobilenet_v2",
    "clahe": false
  }
  ```
  or
  ```json
  {
    "image_base64": "data:image/jpeg;base64,...",
    "model": "mobilenet_v2"
  }
  ```

**Sample Response:**
```json
{
  "status": "success",
  "metadata": {
    "width": 1024,
    "height": 1024,
    "channels": 3,
    "aspect_ratio": 1.0,
    "file_size": "796.2 KB",
    "brightness": 138.4,
    "contrast": 54.2
  },
  "prediction": {
    "model_name": "MobileNetV2",
    "inference_time_ms": 28.4,
    "top_prediction": {
      "rank": 1,
      "class_id": "n02099712",
      "label": "Golden Retriever",
      "probability": 94.85,
      "confidence_level": "High"
    },
    "predictions": [ ... ]
  },
  "cv_diagnostics": {
    "gradcam_heatmap": "data:image/jpeg;base64,...",
    "edges_canny": "data:image/png;base64,...",
    "contours_hud": "data:image/jpeg;base64,...",
    "color_histogram": "data:image/png;base64,..."
  }
}
```

---

## 🛠️ Project Structure

```
├── app.py                   # Flask web server and REST API endpoints
├── model_engine.py          # TensorFlow/Keras inference engine & Grad-CAM
├── cv_pipeline.py           # OpenCV preprocessing, Canny edges & contours
├── train_custom_model.py    # Custom model training script
├── requirements.txt         # Project dependencies
├── templates/
│   └── index.html           # Modern glassmorphic frontend UI
├── static/
│   ├── css/
│   │   └── style.css        # Vanilla CSS design tokens & animations
│   ├── js/
│   │   └── main.js          # Client-side dynamic interaction controller
│   └── samples/             # Benchmark test images (golden retriever, etc.)
└── README.md                # System documentation
```
