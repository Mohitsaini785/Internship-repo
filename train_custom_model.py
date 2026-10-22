"""
Custom Image Classifier Training Script
Uses TensorFlow / Keras Transfer Learning (MobileNetV2 / ResNet50)
to train a custom object classifier on user-provided folders.

Usage:
    # 1. Generate sample training dataset structure:
    py train_custom_model.py --generate_samples

    # 2. Train on your dataset folder (e.g. data/ with subfolders per class):
    py train_custom_model.py --data_dir data/ --epochs 10 --batch_size 16
"""

import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_transfer_model(num_classes: int, img_size=(224, 224), base_arch="mobilenet_v2"):
    """Build a modern transfer learning classification model."""
    inputs = layers.Input(shape=(*img_size, 3))

    # Data Augmentation pipeline
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    x = data_augmentation(inputs)

    # Base Pretrained Feature Extractor
    if base_arch == "resnet50":
        x = keras.applications.resnet50.preprocess_input(x)
        base_model = keras.applications.ResNet50(
            input_shape=(*img_size, 3),
            include_top=False,
            weights="imagenet"
        )
    else:
        x = keras.applications.mobilenet_v2.preprocess_input(x)
        base_model = keras.applications.MobileNetV2(
            input_shape=(*img_size, 3),
            include_top=False,
            weights="imagenet"
        )

    base_model.trainable = False  # Freeze base layers for initial feature extraction
    x = base_model(x, training=False)

    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"Custom_{base_arch}")
    return model, base_model


def train(data_dir: str, epochs: int = 10, batch_size: int = 16, lr: float = 1e-3, output_model: str = "custom_model.keras"):
    """Load dataset, train model, and save weights."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found.")

    img_size = (224, 224)
    print(f"Loading training data from: {data_dir}")

    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Optimize datasets for IO performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build model
    model, base_model = build_transfer_model(num_classes=num_classes, img_size=img_size)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
    ]

    print("\n--- Phase 1: Training Classification Head ---")
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Optional Phase 2: Fine-tuning top layers of the base model
    print("\n--- Phase 2: Fine-Tuning Top Convolutional Layers ---")
    base_model.trainable = True
    # Freeze bottom layers, fine-tune the top 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr / 10),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_phase2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max(2, epochs // 2),
        callbacks=callbacks
    )

    # Save trained model & labels
    model.save(output_model)
    labels_file = output_model.replace(".keras", "_labels.txt")
    with open(labels_file, "w") as f:
        f.write("\n".join(class_names))

    print(f"\nModel successfully saved to: {output_model}")
    print(f"Class labels saved to: {labels_file}")
    return model, class_names


def generate_sample_dataset(dest_dir: str = "dataset_sample"):
    """Create a sample multi-class dataset using OpenCV for immediate experimentation."""
    os.makedirs(dest_dir, exist_ok=True)
    categories = ["circles", "rectangles", "triangles"]

    print(f"Generating synthetic OpenCV image dataset in '{dest_dir}'...")
    for cat in categories:
        cat_dir = os.path.join(dest_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)

        for i in range(25):
            img = np.full((224, 224, 3), 240, dtype=np.uint8)
            color = tuple(int(c) for c in np.random.randint(40, 220, size=3))

            if cat == "circles":
                radius = np.random.randint(30, 70)
                center = (np.random.randint(80, 144), np.random.randint(80, 144))
                cv2.circle(img, center, radius, color, -1)
            elif cat == "rectangles":
                pt1 = (np.random.randint(20, 80), np.random.randint(20, 80))
                pt2 = (np.random.randint(130, 200), np.random.randint(130, 200))
                cv2.rectangle(img, pt1, pt2, color, -1)
            elif cat == "triangles":
                pts = np.array([
                    [np.random.randint(80, 144), np.random.randint(20, 60)],
                    [np.random.randint(20, 70), np.random.randint(150, 200)],
                    [np.random.randint(150, 200), np.random.randint(150, 200)]
                ], np.int32)
                cv2.fillPoly(img, [pts], color)

            # Add subtle noise
            noise = np.random.normal(0, 10, img.shape).astype(np.int16)
            noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(cat_dir, f"sample_{i+1:02d}.jpg"), noisy_img)

    print(f"Done! Created 75 synthetic images across 3 classes in '{dest_dir}'.")
    print(f"You can train on it now with: py train_custom_model.py --data_dir {dest_dir} --epochs 3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom Image Classification model with TensorFlow & OpenCV")
    parser.add_argument("--data_dir", type=str, default="dataset_sample", help="Path to dataset root folder")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output_model", type=str, default="custom_model.keras", help="Output model path")
    parser.add_argument("--generate_samples", action="store_true", help="Generate synthetic sample dataset")

    args = parser.parse_args()

    if args.generate_samples:
        generate_sample_dataset(args.data_dir)
    else:
        train(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            output_model=args.output_model
        )
