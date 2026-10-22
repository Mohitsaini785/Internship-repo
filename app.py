"""
Image Recognition System - Flask Web Application
Provides real-time object classification, OpenCV computer vision analytics,
webcam snapshot capture, and Grad-CAM neural attention heatmaps.
"""

import base64
import os
import traceback
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import cv_pipeline
from model_engine import engine

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}

UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Render the primary dashboard."""
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def get_models():
    """Return available neural architectures."""
    models_info = [
        {
            "id": key,
            "name": val["name"],
            "description": val["desc"],
            "input_size": f"{val['input_size'][0]}x{val['input_size'][1]} px",
        }
        for key, val in engine.AVAILABLE_MODELS.items()
    ]
    return jsonify({"status": "success", "models": models_info})


@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Return all 1,000 ImageNet recognizable categories."""
    try:
        import json
        import tensorflow as tf
        path = tf.keras.utils.get_file(
            "imagenet_class_index.json",
            "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
            cache_subdir="models"
        )
        with open(path) as f:
            raw = json.loads(f.read())
        
        # Sort and format cleanly
        formatted = [
            {
                "id": raw[str(i)][0],
                "label": raw[str(i)][1].replace("_", " ").title(),
                "raw": raw[str(i)][1]
            }
            for i in range(len(raw))
        ]
        return jsonify({"status": "success", "total": len(formatted), "classes": formatted})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """List bundled high-resolution test sample images."""
    samples = [
        {
            "id": "golden_retriever",
            "name": "Golden Retriever Dog",
            "category": "Animals / Canines",
            "url": "/static/samples/golden_retriever.jpg",
            "thumbnail": "/static/samples/golden_retriever.jpg"
        },
        {
            "id": "sports_car",
            "name": "Supercar / Sports Car",
            "category": "Vehicles / Transport",
            "url": "/static/samples/sports_car.jpg",
            "thumbnail": "/static/samples/sports_car.jpg"
        },
        {
            "id": "monarch_butterfly",
            "name": "Monarch Butterfly",
            "category": "Insects / Nature",
            "url": "/static/samples/monarch_butterfly.jpg",
            "thumbnail": "/static/samples/monarch_butterfly.jpg"
        },
        {
            "id": "coffee_cup",
            "name": "Espresso / Cup & Saucer",
            "category": "Food & Drink",
            "url": "/static/samples/coffee_cup.jpg",
            "thumbnail": "/static/samples/coffee_cup.jpg"
        }
    ]
    return jsonify({"status": "success", "samples": samples})


@app.route("/api/classify", methods=["POST"])
def classify():
    """
    Main classification and OpenCV analysis endpoint.
    Accepts multipart file or JSON with base64 image data.
    """
    try:
        image_bytes = None
        file_size_bytes = 0
        model_key = "mobilenet_v2"
        clahe = False
        sharpen = False
        denoise = False

        if request.is_json:
            data = request.get_json() or {}
            model_key = data.get("model", "mobilenet_v2")
            clahe = bool(data.get("clahe", False))
            sharpen = bool(data.get("sharpen", False))
            denoise = bool(data.get("denoise", False))

            # Sample image selection or webcam base64
            if "sample_id" in data:
                sample_file = os.path.join(app.root_path, "static", "samples", f"{data['sample_id']}.jpg")
                if os.path.exists(sample_file):
                    with open(sample_file, "rb") as f:
                        image_bytes = f.read()
                        file_size_bytes = len(image_bytes)
                else:
                    return jsonify({"error": f"Sample image '{data['sample_id']}' not found"}), 404
            elif "image_base64" in data:
                b64_content = data["image_base64"]
                if "," in b64_content:
                    b64_content = b64_content.split(",", 1)[1]
                image_bytes = base64.b64decode(b64_content)
                file_size_bytes = len(image_bytes)

        elif "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400
            if not allowed_file(file.filename):
                return jsonify({"error": "File type not supported. Use PNG, JPG, WEBP, or BMP."}), 400

            image_bytes = file.read()
            file_size_bytes = len(image_bytes)
            model_key = request.form.get("model", "mobilenet_v2")
            clahe = request.form.get("clahe", "false").lower() == "true"
            sharpen = request.form.get("sharpen", "false").lower() == "true"
            denoise = request.form.get("denoise", "false").lower() == "true"

        if not image_bytes:
            return jsonify({"error": "No image data provided. Upload a file or capture a photo."}), 400

        # Step 1: Decode using OpenCV & PIL fallback with EXIF handling
        bgr_img = cv_pipeline.decode_image_bytes(image_bytes)
        metadata = cv_pipeline.get_image_metadata(bgr_img, file_size_bytes=file_size_bytes)

        # Step 2: Apply optional OpenCV enhancements
        processed_img = cv_pipeline.apply_enhancements(
            bgr_img, clahe=clahe, sharpen=sharpen, denoise=denoise
        )

        # Step 3: Run OpenCV Computer Vision Analytics
        edge_visualization = cv_pipeline.detect_edges_canny(processed_img)
        contour_visualization, detections = cv_pipeline.detect_contours_and_boxes(processed_img)
        hist_visualization, hist_data = cv_pipeline.compute_histogram_visualization(processed_img)

        # Step 4: Run TensorFlow / Keras Deep Learning Prediction & Grad-CAM
        ai_results = engine.predict(
            cv_img_bgr=processed_img,
            model_key=model_key,
            top_k=5,
            generate_gradcam=True
        )

        # Base64 of the processed image
        processed_preview = cv_pipeline.encode_image_base64(processed_img, fmt="jpeg")

        return jsonify({
            "status": "success",
            "metadata": metadata,
            "prediction": ai_results,
            "cv_diagnostics": {
                "processed_preview": processed_preview,
                "gradcam_heatmap": ai_results.get("gradcam_heatmap"),
                "edges_canny": edge_visualization,
                "contours_hud": contour_visualization,
                "color_histogram": hist_visualization,
                "histogram_data": hist_data,
                "detected_regions": detections
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "models_cached": list(engine._models.keys()),
        "service": "Image Recognition System"
    })


if __name__ == "__main__":
    print("\n========================================================")
    print(">> Image Recognition System running on http://127.0.0.1:5000")
    print("   Models: MobileNetV2 (Cached), ResNet-50 (Cached)")
    print("   Computer Vision: OpenCV 5.0")
    print("========================================================\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
