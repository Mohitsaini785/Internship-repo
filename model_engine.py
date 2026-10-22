"""
TensorFlow / Keras Model Engine
Manages model loading (MobileNetV2, ResNet50), preprocessing,
fast inference, top-5 prediction decoding, and Grad-CAM visual explanations.
"""

import os
import time
import cv2
import numpy as np

# Suppress verbose TensorFlow C++ logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from cv_pipeline import encode_image_base64


class ModelEngine:
    """Manages Deep Learning image recognition models."""

    AVAILABLE_MODELS = {
        "mobilenet_v2": {
            "name": "MobileNetV2",
            "desc": "Ultra-fast, lightweight neural network optimized for real-time mobile and edge vision.",
            "target_layer": "out_relu",
            "input_size": (224, 224),
            "preprocess_fn": tf.keras.applications.mobilenet_v2.preprocess_input,
            "decode_fn": tf.keras.applications.mobilenet_v2.decode_predictions,
            "loader": tf.keras.applications.MobileNetV2,
        },
        "resnet50": {
            "name": "ResNet-50",
            "desc": "50-layer deep residual network with skip connections for high-accuracy feature extraction.",
            "target_layer": "conv5_block3_out",
            "input_size": (224, 224),
            "preprocess_fn": tf.keras.applications.resnet50.preprocess_input,
            "decode_fn": tf.keras.applications.resnet50.decode_predictions,
            "loader": tf.keras.applications.ResNet50,
        },
    }

    def __init__(self):
        self._models: dict[str, tf.keras.Model] = {}
        self._grad_models: dict[str, tf.keras.Model] = {}
        # Pre-load MobileNetV2 on startup for instant response
        self.get_model("mobilenet_v2")

    def get_model(self, model_key: str = "mobilenet_v2") -> tf.keras.Model:
        """Fetch or initialize the specified model with cached weights."""
        if model_key not in self.AVAILABLE_MODELS:
            model_key = "mobilenet_v2"

        if model_key not in self._models:
            config = self.AVAILABLE_MODELS[model_key]
            print(f"[ModelEngine] Loading {config['name']} with ImageNet weights...")
            model = config["loader"](weights="imagenet")
            self._models[model_key] = model

            # Build Grad-CAM sub-model
            target_layer = config["target_layer"]
            try:
                grad_model = tf.keras.models.Model(
                    inputs=model.inputs,
                    outputs=[model.get_layer(target_layer).output, model.output]
                )
                self._grad_models[model_key] = grad_model
            except Exception as e:
                print(f"[ModelEngine] Grad-CAM setup warning for {model_key}: {e}")

            # Warm-up inference
            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            prep = config["preprocess_fn"](dummy)
            _ = model.predict(prep, verbose=0)
            print(f"[ModelEngine] {config['name']} warmed up and ready.")

        return self._models[model_key]

    def predict(
        self,
        cv_img_bgr: np.ndarray,
        model_key: str = "mobilenet_v2",
        top_k: int = 5,
        generate_gradcam: bool = True
    ) -> dict:
        """
        Run inference on an OpenCV BGR image and return predictions with diagnostics.
        """
        if model_key not in self.AVAILABLE_MODELS:
            model_key = "mobilenet_v2"

        config = self.AVAILABLE_MODELS[model_key]
        model = self.get_model(model_key)

        # 1. Preprocessing with OpenCV & Keras (Aspect-Ratio Preserving Center Crop)
        orig_h, orig_w = cv_img_bgr.shape[:2]
        
        # Convert BGR -> RGB for TensorFlow model
        rgb_img = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2RGB)
        
        # Scale shorter side and center-crop to target 224x224 (Standard ImageNet Protocol)
        # to eliminate horizontal/vertical geometric distortion
        scale = max(config["input_size"][0] / orig_w, config["input_size"][1] / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        scaled_img = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        start_x = max(0, (new_w - config["input_size"][0]) // 2)
        start_y = max(0, (new_h - config["input_size"][1]) // 2)
        cropped_rgb = scaled_img[start_y:start_y + config["input_size"][1], start_x:start_x + config["input_size"][0]]
        
        if cropped_rgb.shape[:2] != config["input_size"]:
            cropped_rgb = cv2.resize(cropped_rgb, config["input_size"])

        img_array = np.expand_dims(cropped_rgb.astype(np.float32), axis=0)
        preprocessed = config["preprocess_fn"](img_array)

        # 2. Timing inference
        start_time = time.perf_counter()
        raw_preds = model.predict(preprocessed, verbose=0)
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # 3. Decode predictions
        decoded = config["decode_fn"](raw_preds, top=top_k)[0]
        
        predictions = []
        for rank, (class_id, class_name, score) in enumerate(decoded, start=1):
            prob = float(score) * 100
            readable_label = class_name.replace("_", " ").title()
            predictions.append({
                "rank": rank,
                "class_id": class_id,
                "label": readable_label,
                "raw_label": class_name,
                "probability": round(prob, 2),
                "confidence_level": self._classify_confidence(prob)
            })

        # 4. Compute Grad-CAM Attention Heatmap
        gradcam_b64 = None
        if generate_gradcam and model_key in self._grad_models:
            try:
                top_class_idx = int(np.argmax(raw_preds[0]))
                gradcam_b64 = self._compute_gradcam(
                    model_key=model_key,
                    preprocessed_tensor=preprocessed,
                    cv_img_bgr=cv_img_bgr,
                    target_class_idx=top_class_idx
                )
            except Exception as e:
                print(f"[ModelEngine] Grad-CAM computation error: {e}")

        return {
            "model_key": model_key,
            "model_name": config["name"],
            "inference_time_ms": inference_time_ms,
            "top_prediction": predictions[0] if predictions else None,
            "predictions": predictions,
            "gradcam_heatmap": gradcam_b64
        }

    def _compute_gradcam(
        self,
        model_key: str,
        preprocessed_tensor: np.ndarray,
        cv_img_bgr: np.ndarray,
        target_class_idx: int
    ) -> str:
        """Generate Grad-CAM activation heatmap overlaid on the original image."""
        grad_model = self._grad_models[model_key]
        
        tensor_in = tf.convert_to_tensor(preprocessed_tensor)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(tensor_in)
            loss = predictions[:, target_class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the channels by gradient importance
        cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
        cam = np.maximum(cam.numpy(), 0)
        
        # Normalize between 0 and 1
        max_val = np.max(cam)
        if max_val > 0:
            cam = cam / max_val
        else:
            cam = np.zeros_like(cam)

        # Resize heatmap back to match original image dimensions
        orig_h, orig_w = cv_img_bgr.shape[:2]
        cam_resized = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        
        # Colorize with JET colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

        # Blend with original BGR image
        blended = cv2.addWeighted(cv_img_bgr, 0.65, heatmap, 0.35, 0)

        return encode_image_base64(blended, fmt="jpeg", quality=85)

    @staticmethod
    def _classify_confidence(prob: float) -> str:
        if prob >= 75.0:
            return "High"
        elif prob >= 35.0:
            return "Moderate"
        else:
            return "Low"


# Global singleton instance
engine = ModelEngine()
