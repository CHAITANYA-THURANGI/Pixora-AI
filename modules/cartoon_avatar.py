"""
Runtime dataset avatar matcher & expression animator (v2.2).

Loads the 30,000-avatar index built from Google's Cartoon Set (CC BY 4.0)
located in datasets/cartoonset100k_jpg/. Given a user's face in the live video:

  1. FEATURE MATCHING: Finds the authentic dataset cartoon avatar that best
     matches the user's hair color, skin tone, and facial proportions.
  2. LIVE EXPRESSION ANIMATION: Dynamically modifies the matched dataset
     avatar in real time to mirror the user's live facial expressions:
       - Blinks & winks (eyeBlinkLeft, eyeBlinkRight)
       - Eye gaze direction (eyeLookUp/Down/In/Out)
       - Eyebrows raise & furrow (browInnerUp, browDownLeft/Right)
       - Mouth smile, frown, open/talk, pucker (jawOpen, mouthSmile, mouthFrown, mouthPucker)
       - Teeth and tongue hints when mouth opens wide
  3. COMPOSITING: Warps, scales, and alpha-blends the animated dataset avatar
     seamlessly onto the user's face position and head rotation in the webcam video.
"""

import json
import os
import time

import cv2
import numpy as np

import config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "models", "cartoon_avatar_index")
FEATURES_DEEP_PATH = os.path.join(INDEX_DIR, "features_deep.npy")
ATTRIBUTES_PATH = os.path.join(INDEX_DIR, "attributes.npy")
FEATURES_PATH = os.path.join(INDEX_DIR, "features.npy")
PATHS_PATH = os.path.join(INDEX_DIR, "paths.json")

_deep_features = None
_attributes = None
_paths = None
_data_dir = None
_last_match_idx = -1
_deep_model = None
_deep_device = None
_avatar_cache = {}  # caches loaded avatar images from disk
_face_match_cache = {}  # face_id -> (matched_avatar_img, path, timestamp)


def avatar_index_available() -> bool:
    """Returns True if the pre-built cartoon avatar index files exist."""
    return (os.path.exists(FEATURES_DEEP_PATH) or os.path.exists(FEATURES_PATH)) and os.path.exists(PATHS_PATH)


def _load_deep_model():
    """Lazy loads MobileNetV3 deep vision feature extractor on GPU."""
    global _deep_model, _deep_device
    if _deep_model is not None:
        return _deep_model
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models
        _deep_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        base = models.mobilenet_v3_large(weights=weights)
        _deep_model = nn.Sequential(*list(base.children())[:-1], nn.Flatten()).to(_deep_device)
        _deep_model.eval()
        return _deep_model
    except Exception as e:
        print(f"[cartoon_avatar] Deep vision model fallback: {e}")
        return None


def _load_index():
    """Loads the pre-built 30,000-avatar deep neural matrix, attributes, and paths."""
    global _deep_features, _attributes, _paths, _data_dir
    if _paths is not None:
        return
    if os.path.exists(FEATURES_DEEP_PATH):
        _deep_features = np.load(FEATURES_DEEP_PATH)
    if os.path.exists(ATTRIBUTES_PATH):
        _attributes = np.load(ATTRIBUTES_PATH)
    elif os.path.exists(FEATURES_PATH):
        _attributes = np.load(FEATURES_PATH)

    with open(PATHS_PATH) as f:
        meta = json.load(f)
    _data_dir = meta.get("data_dir", os.path.join(BASE_DIR, "datasets", "cartoonset100k_jpg"))
    if not os.path.isdir(_data_dir):
        _data_dir = os.path.join(BASE_DIR, "datasets", "cartoonset100k_jpg")
    _paths = meta["paths"]
    print(f"[cartoon_avatar] Loaded {len(_paths)}-avatar deep dataset index from {_data_dir}.")


def _extract_semantic_attributes(img_bgr, landmarks=None):
    """
    Extracts 16-dimensional fine-grained perceptual and physical descriptors:
      - Hair color (dominant centroid in LAB)
      - Skin tone (median highlight-free LAB on cheeks/forehead)
      - Beard / facial hair density & texture
      - Geometric proportions (face aspect ratio)
    """
    h, w = img_bgr.shape[:2]
    if landmarks is not None and len(landmarks) >= 468:
        # Landmark-guided precise face sampling
        skin_pts = [
            (int(landmarks[234].x * w), int(landmarks[234].y * h)),  # left cheek
            (int(landmarks[454].x * w), int(landmarks[454].y * h)),  # right cheek
            (int(landmarks[10].x * w), int(landmarks[10].y * h) + int(h * 0.04)),  # forehead
            (int(landmarks[1].x * w), int(landmarks[1].y * h)),  # nose
        ]
        skin_colors = []
        for px, py in skin_pts:
            px = max(2, min(w - 3, px))
            py = max(2, min(h - 3, py))
            patch = img_bgr[py - 2:py + 3, px - 2:px + 3]
            if patch.size > 0:
                skin_colors.append(patch.mean(axis=(0, 1)))
        skin_bgr = np.median(skin_colors, axis=0) if skin_colors else np.array([128.0, 128.0, 128.0])

        forehead_y = int(landmarks[10].y * h)
        hair_y1 = max(0, forehead_y - int(h * 0.22))
        hair_y2 = max(0, forehead_y - int(h * 0.02))
        hair_x1 = max(0, int(landmarks[234].x * w))
        hair_x2 = min(w, int(landmarks[454].x * w))
        hair_patch = img_bgr[hair_y1:hair_y2, hair_x1:hair_x2]
        hair_bgr = hair_patch.mean(axis=(0, 1)) if hair_patch.size > 0 else np.array([40.0, 40.0, 40.0])

        chin_y = int(landmarks[152].y * h)
        beard_y1 = max(0, chin_y - int(h * 0.08))
        beard_y2 = min(h, chin_y + int(h * 0.04))
        beard_patch = img_bgr[beard_y1:beard_y2, hair_x1:hair_x2]
        beard_bgr = beard_patch.mean(axis=(0, 1)) if beard_patch.size > 0 else skin_bgr
        beard_diff = np.linalg.norm(skin_bgr - beard_bgr)

        face_w = abs(landmarks[454].x - landmarks[234].x) * w
        face_h = abs(landmarks[152].y - landmarks[10].y) * h
        aspect = float(face_w / max(10.0, face_h))
    else:
        hair_region = img_bgr[int(h * 0.05):int(h * 0.22), int(w * 0.20):int(w * 0.80)]
        skin_region = img_bgr[int(h * 0.40):int(h * 0.60), int(w * 0.35):int(w * 0.65)]
        chin_region = img_bgr[int(h * 0.70):int(h * 0.90), int(w * 0.30):int(w * 0.70)]

        hair_bgr = hair_region.mean(axis=(0, 1)) if hair_region.size > 0 else np.array([40.0, 40.0, 40.0])
        skin_bgr = np.median(skin_region, axis=(0, 1)) if skin_region.size > 0 else np.array([128.0, 128.0, 128.0])
        chin_bgr = chin_region.mean(axis=(0, 1)) if chin_region.size > 0 else skin_bgr
        beard_diff = np.linalg.norm(skin_bgr - chin_bgr)
        aspect = float(w / max(1.0, h))

    skin_lab = cv2.cvtColor(np.uint8([[skin_bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    hair_lab = cv2.cvtColor(np.uint8([[hair_bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

    return np.array([
        hair_lab[0], hair_lab[1], hair_lab[2],          # 0-2: Hair LAB
        skin_lab[0], skin_lab[1], skin_lab[2],          # 3-5: Skin LAB
        beard_diff * 1.5, 20.0,                         # 6-7: Beard
        aspect * 50.0,                                  # 8: Aspect ratio
        skin_lab[1] - skin_lab[2],                      # 9: Chromatic bias
        hair_lab[0] - skin_lab[0],                      # 10: Contrast
        15.0, 10.0,                                     # 11-12: Variance
        skin_lab[0] / max(1.0, hair_lab[0]),            # 13: Luminance ratio
        np.clip(beard_diff, 0, 100),                    # 14: Raw beard
        128.0 - skin_lab[0]                             # 15: Melanin darkness
    ], dtype=np.float32)


def match_avatar(face_crop_bgr, landmarks=None, avoid_repeat=False):
    """
    Finds the authentic matching cartoon avatar from Google's Cartoon Set (30k avatars)
    using Hybrid Deep Vision Embeddings (MobileNetV3 960-D) + 16-D Semantic Attributes.
    """
    global _last_match_idx

    if not avatar_index_available():
        return None, None
    _load_index()

    h, w = face_crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return None, None

    query_attr = _extract_semantic_attributes(face_crop_bgr, landmarks=landmarks)

    # 1. Check if Deep 960-D Vision Embeddings are available
    if _deep_features is not None and len(_deep_features) == len(_paths):
        model = _load_deep_model()
        if model is not None:
            try:
                import torch
                import torch.nn as nn
                import torchvision.transforms as transforms
                from PIL import Image

                tf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                tensor = tf(Image.fromarray(face_rgb)).unsqueeze(0).to(_deep_device)
                with torch.no_grad():
                    q_deep = model(tensor)
                    q_deep = nn.functional.normalize(q_deep, p=2, dim=1).cpu().numpy()[0]

                # Cosine distance on 960-D deep vision space
                deep_dist = 1.0 - np.dot(_deep_features, q_deep)

                # Semantic attribute distances
                hair_dist = np.linalg.norm(_attributes[:, 0:3] - query_attr[0:3], axis=1)
                skin_dist = np.linalg.norm(_attributes[:, 3:6] - query_attr[3:6], axis=1)
                beard_dist = np.abs(_attributes[:, 6] - query_attr[6])
                aspect_dist = np.abs(_attributes[:, 8] - query_attr[8])

                # Hybrid multi-modal ranking
                total_dists = (
                    deep_dist * 0.45 +
                    (skin_dist / 40.0) * 0.30 +
                    (hair_dist / 40.0) * 0.15 +
                    (beard_dist / 50.0) * 0.06 +
                    (aspect_dist / 30.0) * 0.04
                )
            except Exception:
                total_dists = np.linalg.norm(_attributes[:, :len(query_attr)] - query_attr, axis=1)
        else:
            total_dists = np.linalg.norm(_attributes[:, :len(query_attr)] - query_attr, axis=1)
    else:
        # Fallback to semantic attribute distance
        total_dists = np.linalg.norm(_attributes[:, :len(query_attr)] - query_attr, axis=1)

    if avoid_repeat and len(total_dists) > 1:
        dists_for_pick = total_dists.copy()
        if _last_match_idx >= 0:
            dists_for_pick[_last_match_idx] = np.inf
        idx = int(np.argmin(dists_for_pick))
    else:
        idx = int(np.argmin(total_dists))

    _last_match_idx = idx
    rel_path = _paths[idx]

    if rel_path in _avatar_cache:
        return _avatar_cache[rel_path], rel_path

    avatar_path = os.path.join(_data_dir, rel_path)
    if not os.path.exists(avatar_path):
        alt_path = os.path.join(BASE_DIR, "datasets", "cartoonset100k_jpg", rel_path)
        if os.path.exists(alt_path):
            avatar_path = alt_path

    avatar_img = cv2.imread(avatar_path)
    if avatar_img is not None:
        _avatar_cache[rel_path] = avatar_img
    return avatar_img, avatar_path


def animate_dataset_avatar(avatar_img, blendshapes):
    """
    Animates a Google Cartoon Set avatar image according to the user's
    live MediaPipe blendshapes (eyes, eyebrows, mouth, gaze, jaw).

    Returns (animated_bgr, alpha_mask).
    """
    anim = avatar_img.copy()
    h, w = anim.shape[:2]  # typically 500x500

    # Sample skin color from nose/cheek area of the avatar (between eyes and mouth)
    skin_color = [int(c) for c in anim[int(h * 0.58), int(w * 0.50)]]

    bs = blendshapes or {}
    blink_l = bs.get("eyeBlinkLeft", 0.0)
    blink_r = bs.get("eyeBlinkRight", 0.0)
    smile_l = bs.get("mouthSmileLeft", 0.0)
    smile_r = bs.get("mouthSmileRight", 0.0)
    smile = (smile_l + smile_r) / 2.0
    frown_l = bs.get("mouthFrownLeft", 0.0)
    frown_r = bs.get("mouthFrownRight", 0.0)
    frown = (frown_l + frown_r) / 2.0
    jaw_open = bs.get("jawOpen", 0.0)
    pucker = bs.get("mouthPucker", 0.0)
    brow_up = bs.get("browInnerUp", 0.0)
    brow_down = (bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)) / 2.0

    # 1. EYES & GAZE (Canonical Left Eye=(198, 251), Right Eye=(297, 251))
    lex, ley = int(w * 0.397), int(h * 0.502)  # Left Eye center (198, 251)
    rex, rey = int(w * 0.594), int(h * 0.502)  # Right Eye center (297, 251)
    eye_radius_x, eye_radius_y = int(w * 0.055), int(h * 0.045)

    # Gaze offset calculation
    look_in_l = bs.get("eyeLookInLeft", 0.0)
    look_out_l = bs.get("eyeLookOutLeft", 0.0)
    look_up = (bs.get("eyeLookUpLeft", 0.0) + bs.get("eyeLookUpRight", 0.0)) / 2.0
    look_down = (bs.get("eyeLookDownLeft", 0.0) + bs.get("eyeLookDownRight", 0.0)) / 2.0
    gaze_x = int((look_in_l - look_out_l) * w * 0.014)
    gaze_y = int((look_down - look_up) * h * 0.012)

    # Left Eye: Blink / Wink
    if blink_l > 0.35:
        cv2.ellipse(anim, (lex, ley), (eye_radius_x + 2, eye_radius_y + 2), 0, 0, 360, skin_color, -1)
        cv2.ellipse(anim, (lex, ley + 1), (eye_radius_x, int(eye_radius_y * 0.6)), 0, 190, 350, (30, 30, 30), 4, cv2.LINE_AA)
    elif abs(gaze_x) > 1 or abs(gaze_y) > 1:
        cv2.circle(anim, (lex + gaze_x, ley + gaze_y), int(w * 0.010), (255, 255, 255), -1, cv2.LINE_AA)

    # Right Eye: Blink / Wink
    if blink_r > 0.35:
        cv2.ellipse(anim, (rex, rey), (eye_radius_x + 2, eye_radius_y + 2), 0, 0, 360, skin_color, -1)
        cv2.ellipse(anim, (rex, rey + 1), (eye_radius_x, int(eye_radius_y * 0.6)), 0, 190, 350, (30, 30, 30), 4, cv2.LINE_AA)
    elif abs(gaze_x) > 1 or abs(gaze_y) > 1:
        cv2.circle(anim, (rex + gaze_x, rey + gaze_y), int(w * 0.010), (255, 255, 255), -1, cv2.LINE_AA)

    # 2. EYEBROWS (Canonical y=220)
    lbx, lby = int(w * 0.397), int(h * 0.440)
    rbx, rby = int(w * 0.594), int(h * 0.440)
    brow_offset = int((-brow_up * 10 + brow_down * 8) * (h / 500.0))
    if abs(brow_offset) > 3:
        cv2.line(anim, (lbx - 18, lby + brow_offset), (lbx + 18, lby + brow_offset), (40, 40, 40), 3, cv2.LINE_AA)
        cv2.line(anim, (rbx - 18, rby + brow_offset), (rbx + 18, rby + brow_offset), (40, 40, 40), 3, cv2.LINE_AA)

    # 3. MOUTH & JAW (Canonical Mouth=(248, 338))
    mcx, mcy = int(w * 0.496), int(h * 0.676)
    mouth_w = int(w * 0.11)

    if jaw_open > 0.18:
        # Open mouth (talking / shouting / laughing)
        cv2.ellipse(anim, (mcx, mcy), (mouth_w + 8, int(h * 0.065)), 0, 0, 360, skin_color, -1)
        open_h = int(10 + jaw_open * 30)
        open_w = int(mouth_w + smile * 15)
        # Dark mouth interior
        cv2.ellipse(anim, (mcx, mcy), (open_w, open_h), 0, 0, 360, (25, 20, 35), -1, cv2.LINE_AA)
        cv2.ellipse(anim, (mcx, mcy), (open_w, open_h), 0, 0, 360, (30, 30, 30), 3, cv2.LINE_AA)
        # Teeth
        if jaw_open > 0.25:
            teeth_top = mcy - open_h + 3
            teeth_h = min(open_h // 3, 8)
            cv2.rectangle(anim, (mcx - open_w + 6, teeth_top), (mcx + open_w - 6, teeth_top + teeth_h), (250, 250, 250), -1)
        # Tongue
        if jaw_open > 0.35:
            cv2.ellipse(anim, (mcx, mcy + open_h - 4), (open_w // 2, open_h // 3), 0, 0, 360, (120, 110, 220), -1, cv2.LINE_AA)
    elif smile > 0.28:
        # Big smile curve
        cv2.ellipse(anim, (mcx, mcy), (mouth_w + 6, int(h * 0.04)), 0, 0, 360, skin_color, -1)
        curve_pts = np.array([
            [mcx - mouth_w - 3, mcy - 3],
            [mcx - mouth_w // 2, mcy + 8],
            [mcx, mcy + 11],
            [mcx + mouth_w // 2, mcy + 8],
            [mcx + mouth_w + 3, mcy - 3]
        ], dtype=np.int32)
        cv2.polylines(anim, [curve_pts], False, (30, 30, 30), 3, cv2.LINE_AA)
    elif frown > 0.28:
        # Frown curve
        cv2.ellipse(anim, (mcx, mcy), (mouth_w + 5, int(h * 0.035)), 0, 0, 360, skin_color, -1)
        curve_pts = np.array([
            [mcx - mouth_w, mcy + 6],
            [mcx - mouth_w // 2, mcy - 3],
            [mcx, mcy - 5],
            [mcx + mouth_w // 2, mcy - 3],
            [mcx + mouth_w, mcy + 6]
        ], dtype=np.int32)
        cv2.polylines(anim, [curve_pts], False, (30, 30, 30), 3, cv2.LINE_AA)
    elif pucker > 0.30:
        # Pucker circle (kiss face)
        cv2.ellipse(anim, (mcx, mcy), (mouth_w, int(h * 0.035)), 0, 0, 360, skin_color, -1)
        cv2.circle(anim, (mcx, mcy), 10, (30, 30, 30), 3, cv2.LINE_AA)
        cv2.circle(anim, (mcx, mcy), 7, (90, 80, 180), -1, cv2.LINE_AA)

    # 4. BACKGROUND SEGMENTATION (removes white background, keeps character)
    gray = cv2.cvtColor(anim, cv2.COLOR_BGR2GRAY)
    mask = (gray < 246).astype(np.uint8) * 255
    # Smooth alpha mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    return anim, mask


def render_live_3d_cartoon_mask(frame, face_info, passthrough_eyes=False, blend_mode="seamless"):
    """
    Renders an Augmented Reality 3D Live Deformable Cartoon Facial Mask:
      1. Matches best cartoon avatar from dataset
      2. Animates facial expressions (jaw, blinks, brows, smiles)
      3. Harmonizes lighting with camera environment
      4. Warps avatar onto 468 3D facial landmarks using dense Delaunay mesh
      5. Blends seamlessly onto live video with Poisson cloning or alpha feathering

    Args:
        frame: full BGR video frame (modified in-place)
        face_info: FaceInfo object with .landmarks, .blendshapes, .bbox, .head_pose

    Returns:
        modified frame with live 3D cartoon mask rendered
    """
    if not avatar_index_available():
        return frame

    h, w = frame.shape[:2]
    x, y, fw, fh = face_info.bbox
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + fw), min(h, y + fh)
    actual_w, actual_h = x2 - x1, y2 - y1

    if actual_w <= 10 or actual_h <= 10:
        return frame

    # Temporal face match caching (caches avatar selection for 1.5s to maintain 60+ FPS)
    face_key = (int(x // 40), int(y // 40))
    now = time.time()
    avatar_img = None

    if face_key in _face_match_cache and (now - _face_match_cache[face_key][1]) < 1.5:
        avatar_img = _face_match_cache[face_key][0]
    else:
        # Crop the face region to match against dataset via Deep Vision Model
        face_crop = frame[y1:y2, x1:x2]
        avatar_img, _ = match_avatar(face_crop, landmarks=face_info.landmarks, avoid_repeat=False)
        if avatar_img is not None:
            _face_match_cache[face_key] = (avatar_img, now)

    if avatar_img is None:
        return frame

    # Animate dataset avatar using live blendshapes
    anim_avatar, anim_mask = animate_dataset_avatar(avatar_img, face_info.blendshapes)

    # Harmonize lighting with camera ambient lighting
    from modules.cartoon_warp_engine import (
        harmonize_lighting, render_snapchat_ar_mask
    )
    ambient_sample = frame[max(0, y - 20):max(1, y), max(0, x):min(w, x + fw)]
    ambient_bgr = np.mean(ambient_sample, axis=(0, 1)) if ambient_sample.size > 0 else np.array([128.0, 128.0, 128.0])
    anim_avatar = harmonize_lighting(anim_avatar, ambient_bgr)

    # Render Snapchat-quality Conformal AR cartoon face mask
    lm = face_info.landmarks
    if lm and len(lm) >= 468:
        return render_snapchat_ar_mask(
            frame, anim_avatar, anim_mask, face_info,
            passthrough_eyes=passthrough_eyes,
            blend_strength=0.95
        )

    # Fallback when landmarks are sparse (bounding box + head pose only)
    cx, cy = x + fw // 2, y + fh // 2
    roll_deg = face_info.head_pose.get("roll", 0.0) if face_info.head_pose else 0.0

    target_w = max(20, int(fw * 1.35))
    target_h = max(20, int(fh * 1.40))
    resized_avatar = cv2.resize(anim_avatar, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    resized_mask = cv2.resize(anim_mask, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    if abs(roll_deg) > 1.5:
        M = cv2.getRotationMatrix2D((target_w / 2, target_h / 2), -roll_deg, 1.0)
        resized_avatar = cv2.warpAffine(
            resized_avatar, M, (target_w, target_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
        )
        resized_mask = cv2.warpAffine(
            resized_mask, M, (target_w, target_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

    x1 = int(cx - target_w / 2)
    y1 = int(cy - target_h * 0.46)
    x2 = x1 + target_w
    y2 = y1 + target_h

    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(w, x2), min(h, y2)

    ax1 = fx1 - x1
    ay1 = fy1 - y1
    ax2 = ax1 + (fx2 - fx1)
    ay2 = ay1 + (fy2 - fy1)

    if fx2 > fx1 and fy2 > fy1 and ax2 > ax1 and ay2 > ay1:
        crop_av = resized_avatar[ay1:ay2, ax1:ax2]
        crop_mask = resized_mask[ay1:ay2, ax1:ax2]

        alpha = (crop_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        bg_roi = frame[fy1:fy2, fx1:fx2].astype(np.float32)
        fg_roi = crop_av.astype(np.float32)

        blended = fg_roi * alpha + bg_roi * (1.0 - alpha)
        frame[fy1:fy2, fx1:fx2] = np.clip(blended, 0, 255).astype(np.uint8)

    return frame


def render_animated_dataset_avatar(frame, face_info):
    """
    Finds the best-matching dataset cartoon avatar for this face,
    animates it in real-time with the user's live expressions,
    and conforms it over the face as a live 3D deformable mask.
    """
    blend_mode = getattr(config, "CARTOON_BLEND_MODE", "seamless")
    passthrough = getattr(config, "CARTOON_PASSTHROUGH_EYES", False)
    return render_live_3d_cartoon_mask(frame, face_info, passthrough_eyes=passthrough, blend_mode=blend_mode)

