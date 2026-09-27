"""
High-Precision Deep Neural Avatar Index Builder for Pixora AI — v3.5.

Leverages PyTorch GPU Deep Vision Backbone (MobileNetV3-Large 960-D) + 16-D Fine-Grained
Perceptual Semantic Descriptors (Hair LAB, Skin LAB, Melanin Index, Beard Density,
Texture Energy, Facial Aspect Ratio) over Google's Cartoon Set (30,000 characters).

Extracts:
  1. 960-D Deep Convolutional Embeddings on GPU (models/cartoon_avatar_index/features_deep.npy)
  2. 16-D Semantic Facial Descriptors (models/cartoon_avatar_index/attributes.npy)
  3. Image Paths Metadata (models/cartoon_avatar_index/paths.json)

Usage:
    venv\\Scripts\\python.exe training/build_cartoon_avatar_index.py --subfolders 0 1 2 --batch-size 64
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "cartoonset100k_jpg")
INDEX_DIR = os.path.join(PROJECT_ROOT, "models", "cartoon_avatar_index")

# Initialize GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vision preprocessing transform
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_feature_model():
    """Loads MobileNetV3-Large on GPU with classification head removed (960-D output)."""
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    base_model = models.mobilenet_v3_large(weights=weights)
    extractor = nn.Sequential(*list(base_model.children())[:-1], nn.Flatten()).to(device)
    extractor.eval()
    return extractor


def extract_semantic_attributes(img_bgr):
    """
    Extracts 16-dimensional fine-grained perceptual and physical descriptors:
      - Hair color in LAB perceptual color space (dominant centroid)
      - Skin tone in LAB perceptual color space (median highlight-free sampling)
      - Beard / facial hair density & Laplacian texture energy
      - Face aspect ratio and geometric proportions
      - Melanin darkness & chromatic warmth
    """
    h, w = img_bgr.shape[:2]
    hair_crop = img_bgr[int(h * 0.05):int(h * 0.22), int(w * 0.20):int(w * 0.80)]
    skin_crop = img_bgr[int(h * 0.40):int(h * 0.60), int(w * 0.35):int(w * 0.65)]
    chin_crop = img_bgr[int(h * 0.70):int(h * 0.90), int(w * 0.30):int(w * 0.70)]

    hair_lab = cv2.cvtColor(hair_crop, cv2.COLOR_BGR2LAB) if hair_crop.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8)
    hair_mean = hair_lab.mean(axis=(0, 1))

    skin_lab = cv2.cvtColor(skin_crop, cv2.COLOR_BGR2LAB) if skin_crop.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8)
    skin_median = np.median(skin_lab, axis=(0, 1))

    chin_lab = cv2.cvtColor(chin_crop, cv2.COLOR_BGR2LAB) if chin_crop.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8)
    chin_mean = chin_lab.mean(axis=(0, 1))
    beard_diff = np.linalg.norm(skin_median - chin_mean)

    chin_gray = cv2.cvtColor(chin_crop, cv2.COLOR_BGR2GRAY) if chin_crop.size > 0 else np.zeros((10, 10), dtype=np.uint8)
    lap_var = cv2.Laplacian(chin_gray, cv2.CV_64F).var()

    aspect = float(w / max(1.0, h))

    return np.array([
        hair_mean[0], hair_mean[1], hair_mean[2],          # Hair L, A, B (0-2)
        skin_median[0], skin_median[1], skin_median[2],    # Skin L, A, B (3-5)
        beard_diff * 1.5, min(100.0, lap_var * 0.5),      # Beard / Texture (6-7)
        aspect * 50.0,                                     # Aspect ratio (8)
        skin_median[1] - skin_median[2],                   # Skin chromatic bias (9)
        hair_mean[0] - skin_median[0],                     # Hair/Skin contrast (10)
        float(np.std(hair_lab[:, :, 0])) if hair_crop.size > 0 else 0.0, # Hair variance (11)
        float(np.std(skin_lab[:, :, 0])) if skin_crop.size > 0 else 0.0, # Skin variance (12)
        skin_median[0] / max(1.0, hair_mean[0]),           # Relative luminance ratio (13)
        np.clip(beard_diff, 0, 100),                       # Raw beard diff (14)
        128.0 - skin_median[0]                             # Melanin darkness (15)
    ], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Build Deep Neural & Semantic Avatar Index")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--subfolders", nargs="+", default=["0", "1", "2"],
                        help="Subfolders to index (default: 0 1 2, ~30k avatars)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="GPU batch size for deep feature extraction")
    parser.add_argument("--max-per-folder", type=int, default=None,
                        help="Cap images per folder for testing")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise RuntimeError(f"Dataset directory not found: {args.data_dir}")

    all_files = []
    for sub in args.subfolders:
        folder = os.path.join(args.data_dir, sub)
        if not os.path.isdir(folder):
            print(f"  Skipping missing subfolder: {folder}")
            continue
        files = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if args.max_per_folder:
            files = files[:args.max_per_folder]
        all_files.extend(files)

    if not all_files:
        raise RuntimeError("No images found in dataset subfolders.")

    print(f"[build_index] Initializing Deep Vision Model on {device}...")
    model = load_feature_model()

    print(f"[build_index] Processing {len(all_files)} avatars in GPU batches of {args.batch_size}...")
    t0 = time.time()

    deep_features = []
    attributes = []
    paths = []

    num_batches = (len(all_files) + args.batch_size - 1) // args.batch_size

    for b in range(num_batches):
        batch_paths = all_files[b * args.batch_size:(b + 1) * args.batch_size]
        batch_imgs_pil = []
        batch_attrs = []
        valid_paths = []

        for p in batch_paths:
            img_bgr = cv2.imread(p)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            batch_imgs_pil.append(Image.fromarray(img_rgb))
            batch_attrs.append(extract_semantic_attributes(img_bgr))
            valid_paths.append(os.path.relpath(p, args.data_dir))

        if not batch_imgs_pil:
            continue

        with torch.no_grad():
            tensors = torch.stack([img_transform(img) for img in batch_imgs_pil]).to(device)
            embeddings = model(tensors)
            norm_embeddings = nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()

        deep_features.append(norm_embeddings)
        attributes.extend(batch_attrs)
        paths.extend(valid_paths)

        if (b + 1) % 50 == 0 or (b + 1) == num_batches:
            processed = len(paths)
            elapsed = time.time() - t0
            rate = processed / max(0.1, elapsed)
            print(f"  Batch {b + 1}/{num_batches} | Processed {processed}/{len(all_files)} ({rate:.1f} img/sec)...")

    deep_features = np.vstack(deep_features).astype(np.float32)
    attributes = np.array(attributes, dtype=np.float32)

    os.makedirs(INDEX_DIR, exist_ok=True)
    # Save deep embeddings (960-D) and semantic attributes (16-D)
    np.save(os.path.join(INDEX_DIR, "features_deep.npy"), deep_features)
    np.save(os.path.join(INDEX_DIR, "attributes.npy"), attributes)
    # Also save unified 16-D features.npy for backward-compatibility
    np.save(os.path.join(INDEX_DIR, "features.npy"), attributes)

    with open(os.path.join(INDEX_DIR, "paths.json"), "w") as f:
        json.dump({"data_dir": args.data_dir, "paths": paths}, f)

    elapsed = time.time() - t0
    print(f"\n✅ Successfully built Deep Neural Index for {len(paths)} avatars in {elapsed:.2f}s ({len(paths)/max(0.1, elapsed):.1f} img/sec) on {device}!")
    print(f"Deep Feature Matrix Shape: {deep_features.shape} (960-D MobileNetV3 Embeddings)")
    print(f"Semantic Attributes Shape: {attributes.shape} (16-D Perceptual LAB Descriptors)")
    print(f"Saved to: {INDEX_DIR}/ (features_deep.npy + attributes.npy + paths.json)")


if __name__ == "__main__":
    main()


