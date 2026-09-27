"""
Cartoonification -- Snapchat-style live face cartoonification (v2.4).

Five modes, set via config.CARTOONIFY_MODE:

  "avatar_match" (DEFAULT, dataset-matched & animated): Matches the closest
      cartoon avatar from Google's Cartoon Set (30,000 indexed characters
      in datasets/cartoonset100k_jpg) to YOUR hair color & skin tone, and
      animates it in real time with YOUR live facial expressions (blinks,
      winks, smiles, talking/jawOpen, gaze tracking, eyebrows).

  "cyber_hologram" (FUTURISTIC): Glowing cyberpunk hologram avatar with
      electric neon contours, cybernetic ocular glowing eyes, scanlines,
      and energy circuit pulse.

  "expression": Snapchat-style 2D vector cartoon face filter drawn directly
      over each detected face mesh, mirroring your exact expressions.

  "gan": AnimeGANv2 (bryandlee/animegan2-pytorch), neural style-transfer
      network -- applies an anime art style to the actual video feed.
      Preserves your appearance in anime form. Runs on GPU.

  "opencv": a fast, classic image-processing filter (bilateral filter +
      adaptive-threshold edge overlay). Real-time on CPU.

All modes degrade gracefully to the OpenCV filter if their requirements
aren't met, so cartoonify() never fails to produce *something*.
"""

import cv2
import numpy as np

import config

_gan_models = {}   # one lazy-loaded singleton per style, keyed by style name
_gan_device = None  # cached torch device for GAN inference

# Official AnimeGANv2 checkpoints (bryandlee/animegan2-pytorch)
GAN_STYLES = {
    "portrait_v2": "face_paint_512_v2",   # clean, modern anime portrait look
    "portrait_v1": "face_paint_512_v1",   # softer, slightly painterly variant
    "celeba_distill": "celeba_distill",   # smoother, more stylized/simplified
    "paprika": "paprika",                 # bolder linework, more graphic-novel-like
}

# MediaPipe face mesh FACE OVAL landmark indices.
# These 36 points trace the outline of the face (jawline, temples, forehead)
# and are used to create a face-shaped mask -- the same technique Snapchat
# and Instagram use for their face filters.
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


# ================================================================ OPENCV ===
def cartoonify_opencv(frame_bgr, num_down=2, num_bilateral=7):
    """Classic image-processing cartoon filter: bilateral smoothing + edge overlay."""
    img = frame_bgr.copy()

    img_color = img
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    img_color = cv2.resize(img_color, (img.shape[1], img.shape[0]))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)
    edges = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2,
    )
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(img_color, edges_colored)


# ==================================================================== GAN ===
def _get_gan_device():
    """Cached torch device -- determines once whether we have CUDA."""
    global _gan_device
    if _gan_device is None:
        import torch
        _gan_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _gan_device


def _load_gan_model(style_key):
    """Lazy singleton load. Model is loaded once per style, moved to GPU."""
    if style_key not in _gan_models:
        import torch
        checkpoint = GAN_STYLES.get(style_key, GAN_STYLES["portrait_v2"])
        device = _get_gan_device()
        print(f"[cartoonify] Loading AnimeGANv2 '{style_key}' ({checkpoint}) on {device}...")
        model = torch.hub.load(
            "bryandlee/animegan2-pytorch:main",
            "generator",
            pretrained=checkpoint,
            trust_repo=True,
        )
        model = model.to(device)
        model.eval()
        _gan_models[style_key] = model
        print(f"[cartoonify] AnimeGANv2 '{style_key}' loaded on {device}.")
    return _gan_models[style_key]


def _gan_inference(frame_bgr, resolution, style=None):
    """Core GAN inference with FP16 autocast on CUDA."""
    import torch
    from PIL import Image

    style = style or config.GAN_STYLE
    model = _load_gan_model(style)
    device = _get_gan_device()

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).resize((resolution, resolution), Image.LANCZOS)

    x = np.asarray(pil_img).astype(np.float32) / 127.5 - 1.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(x)[0]
        else:
            out = model(x)[0]

    out = out.cpu().permute(1, 2, 0).numpy()
    out = np.clip((out + 1.0) * 127.5, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    result = cv2.resize(out_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    # Post-processing: sharpen the GAN output for a crisper cartoon look
    if getattr(config, "CARTOON_SHARPEN", True):
        result = _sharpen(result)

    return result


def _sharpen(image):
    """Subtle unsharp mask: sharpens details without introducing noise.
    Makes the GAN's cartoon lines and colors pop instead of looking washed out."""
    gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(image, 1.4, gaussian, -0.4, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _color_preserve(original_bgr, cartoon_bgr, strength=0.3):
    """Transfers luminance from the cartoon while preserving some of the
    original's color distribution. This keeps natural skin tones visible
    through the cartoon style, making it look more 'fitted' to the person."""
    orig_lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    cart_lab = cv2.cvtColor(cartoon_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Blend the A and B (color) channels: keep cartoon's luminance, mix colors
    cart_lab[:, :, 1] = cart_lab[:, :, 1] * (1 - strength) + orig_lab[:, :, 1] * strength
    cart_lab[:, :, 2] = cart_lab[:, :, 2] * (1 - strength) + orig_lab[:, :, 2] * strength
    return cv2.cvtColor(np.clip(cart_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def cartoonify_gan(frame_bgr, style=None):
    """Full-quality AnimeGANv2 render at 512x512. Used for saved memes."""
    return _gan_inference(frame_bgr, resolution=512, style=style)


def cartoonify_gan_live(frame_bgr, style=None):
    """High-quality GAN render for live preview at config.GAN_LIVE_RESOLUTION
    (default 512 for Snapchat-quality, 256 for faster FPS on slower GPUs)."""
    live_res = getattr(config, "GAN_LIVE_RESOLUTION", 512)
    return _gan_inference(frame_bgr, resolution=live_res, style=style)


# ============================================================ AVATAR MATCH ===
def cartoonify_avatar_match(frame_bgr):
    """Feature-matched avatar from Google's Cartoon Set."""
    from modules.cartoon_avatar import match_avatar, avatar_index_available

    if not avatar_index_available():
        return None
    avatar_img, _path = match_avatar(frame_bgr)
    if avatar_img is None:
        return None
    return cv2.resize(avatar_img, (frame_bgr.shape[1], frame_bgr.shape[0]))


# ====================================================== FACE MASK (v2.1) ===
def create_face_mask_from_landmarks(landmarks, frame_h, frame_w, crop_x, crop_y, crop_w, crop_h):
    """Creates a face-shaped binary mask from MediaPipe's 478 face landmarks.

    Uses the FACE_OVAL_INDICES (36 points tracing jawline, temples, forehead)
    to create a filled polygon mask, then Gaussian-blurs the edges for seamless
    alpha blending. This is exactly how Snapchat/Instagram face filters work —
    the cartoon only appears inside the face shape, not in a hard rectangle.

    Args:
        landmarks: list of MediaPipe NormalizedLandmark objects (478 points)
        frame_h, frame_w: full frame dimensions (landmarks are in 0-1 normalized coords)
        crop_x, crop_y: top-left corner of the face crop in pixel coordinates
        crop_w, crop_h: width and height of the face crop in pixels

    Returns:
        float32 mask array (crop_h, crop_w) with values 0.0-1.0
    """
    if not landmarks or len(landmarks) < max(FACE_OVAL_INDICES) + 1:
        # Not enough landmarks — return a soft elliptical mask as fallback
        return _create_ellipse_mask(crop_h, crop_w)

    # Convert face oval landmarks from normalized frame coords to crop-local pixels
    points = []
    for idx in FACE_OVAL_INDICES:
        lm = landmarks[idx]
        px = int(lm.x * frame_w) - crop_x
        py = int(lm.y * frame_h) - crop_y
        # Clamp to crop boundaries
        px = max(0, min(crop_w - 1, px))
        py = max(0, min(crop_h - 1, py))
        points.append([px, py])

    if len(points) < 10:
        return _create_ellipse_mask(crop_h, crop_w)

    # Create filled polygon mask from face oval points
    pts = np.array(points, dtype=np.int32)
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # Dilate slightly so the mask covers a bit beyond the landmarks
    # (landmarks sit on the face boundary, we want to include a hair-width margin)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Gaussian blur the edges for seamless blending
    blur_size = getattr(config, "FACE_MASK_BLUR", 31)
    blur_size = blur_size | 1  # ensure odd
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    return mask.astype(np.float32) / 255.0


def _create_ellipse_mask(h, w):
    """Fallback: soft elliptical mask when landmarks aren't available.
    Better than a hard rectangle — at least the corners fade out."""
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.42), int(h * 0.46))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    blur_size = getattr(config, "FACE_MASK_BLUR", 31) | 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    return mask.astype(np.float32) / 255.0


def create_feather_mask(h, w, feather_size=None):
    """Simple rectangular gradient mask — used only as a last-resort fallback
    when neither landmarks nor ellipse are appropriate."""
    feather_size = feather_size or getattr(config, "CARTOON_BLEND_FEATHER", 15)
    mask = np.ones((h, w), dtype=np.float32)
    feather = min(feather_size, h // 4, w // 4)
    if feather < 2:
        return mask
    for i in range(feather):
        val = i / feather
        mask[i, :] = np.minimum(mask[i, :], val)
        mask[h - 1 - i, :] = np.minimum(mask[h - 1 - i, :], val)
        mask[:, i] = np.minimum(mask[:, i], val)
        mask[:, w - 1 - i] = np.minimum(mask[:, w - 1 - i], val)
    return mask


# ============================================== SNAPCHAT-STYLE BLENDING ===
def blend_cartoon_face(background, cartoon_crop, x, y, fw, fh, landmarks=None, frame_h=None, frame_w=None):
    """Blends a cartoonified face crop into the background frame using
    a face-shaped mask from MediaPipe landmarks (Snapchat-style).

    If landmarks are provided, creates a precise face-oval mask so the
    cartoon only appears inside the face shape. Falls back to an elliptical
    mask if landmarks aren't available — either way, the result is far more
    natural than a hard rectangular paste.
    """
    h, w = background.shape[:2]
    frame_h = frame_h or h
    frame_w = frame_w or w

    # Clamp to frame boundaries
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + fw), min(h, y + fh)
    actual_w, actual_h = x2 - x1, y2 - y1
    if actual_w <= 4 or actual_h <= 4:
        return background

    # Resize cartoon crop to match the actual region
    resized_crop = cv2.resize(cartoon_crop, (actual_w, actual_h), interpolation=cv2.INTER_LANCZOS4)

    # Color preservation: keep some of the original skin tones
    original_region = background[y1:y2, x1:x2]
    resized_crop = _color_preserve(original_region, resized_crop, strength=0.25)

    # Create the mask — face-shaped if we have landmarks, elliptical otherwise
    if landmarks is not None:
        mask = create_face_mask_from_landmarks(
            landmarks, frame_h, frame_w, x1, y1, actual_w, actual_h
        )
    else:
        mask = _create_ellipse_mask(actual_h, actual_w)

    mask_3ch = mask[:, :, np.newaxis]

    # Alpha blend: cartoon * mask + original * (1 - mask)
    region = background[y1:y2, x1:x2].astype(np.float32)
    blended = resized_crop.astype(np.float32) * mask_3ch + region * (1.0 - mask_3ch)
    background[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return background


# ============================================================ LIVE PREVIEW ===
def cartoonify_live(frame_bgr, mode=None):
    """Live preview cartoonification — respects config.CARTOONIFY_MODE.

    NOTE: "expression" mode is handled at the main.py level (it renders
    cartoon faces directly on the full frame using landmarks), not here.
    If called with mode="expression", falls back to opencv for the crop.

    For "gan" mode, uses config.GAN_LIVE_RESOLUTION (default 512) with
    FP16 autocast for high-quality results on RTX 4060.
    Falls back gracefully to OpenCV if the requested mode fails.
    """
    mode = mode or config.CARTOONIFY_MODE

    if mode == "expression":
        # Expression mode is handled in main.py's apply_multi_face_toon()
        # If we get here (e.g. for saved memes), use opencv as fallback
        return cartoonify_opencv(frame_bgr, num_down=2, num_bilateral=2)

    if mode == "gan":
        try:
            return cartoonify_gan_live(frame_bgr)
        except Exception as e:
            if not getattr(cartoonify_live, "_gan_warned", False):
                print(f"[cartoonify] GAN live preview failed ({e}); falling back to OpenCV filter.")
                cartoonify_live._gan_warned = True
            return cartoonify_opencv(frame_bgr, num_down=2, num_bilateral=2)

    elif mode == "avatar_match":
        try:
            result = cartoonify_avatar_match(frame_bgr)
            if result is not None:
                return result
            if not getattr(cartoonify_live, "_avatar_warned", False):
                print("[cartoonify] Avatar index not built — falling back to OpenCV filter.")
                cartoonify_live._avatar_warned = True
        except Exception:
            pass
        return cartoonify_opencv(frame_bgr, num_down=2, num_bilateral=2)

    # "opencv" mode or unknown
    return cartoonify_opencv(frame_bgr, num_down=2, num_bilateral=2)


# ============================================================= ENTRY POINT ===
def cartoonify(frame_bgr, mode=None):
    """Full-quality cartoonification for saved memes. Never fails outright."""
    mode = mode or config.CARTOONIFY_MODE

    if mode == "avatar_match":
        try:
            result = cartoonify_avatar_match(frame_bgr)
            if result is not None:
                return result
            print("[cartoonify] Avatar index not built — falling back to OpenCV filter.")
        except Exception as e:
            print(f"[cartoonify] Avatar match failed ({e}), falling back to OpenCV filter.")
    elif mode == "gan":
        try:
            return cartoonify_gan(frame_bgr)
        except Exception as e:
            print(f"[cartoonify] GAN path failed ({e}), falling back to OpenCV filter.")

    return cartoonify_opencv(frame_bgr)
