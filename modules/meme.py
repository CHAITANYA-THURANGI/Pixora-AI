"""
Composites a meme-style caption onto an image using PIL.

v2.0 changes:
  - Caption is now at the BOTTOM of the image (matching the live preview bar)
  - Semi-transparent black bar behind the caption for readability on any image
  - Font size scales with min(width, height) so it looks correct on ANY aspect ratio
  - Long captions auto-wrap to multiple lines within the bar
  - Tag/emotion label sits inside the bottom bar as small text
  - Works correctly for portrait, landscape, square, and ultra-wide images

Font loading tries, in order: the font bundled with this project
(assets/fonts/BigShoulders-Bold.ttf, SIL Open Font License -- works on any
OS with zero setup) -> a few common system font paths -> PIL's built-in
default font as a last resort that can never fail. This means make_meme()
always produces a usable image, even on a machine with no fonts configured.
"""

import os
import textwrap

import cv2
from PIL import Image, ImageDraw, ImageFont

import config

_CANDIDATE_FONTS = [
    config.FONT_PATH,
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

# Cache loaded fonts to avoid repeated disk access
_font_cache = {}


def _load_font(size):
    """Load a font at the given size, with fallback chain. Cached."""
    if size in _font_cache:
        return _font_cache[size]
    for path in _CANDIDATE_FONTS:
        if path and os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                _font_cache[size] = font
                return font
            except Exception:
                continue
    font = ImageFont.load_default()  # always works, never raises
    _font_cache[size] = font
    return font


def _wrap_text(draw, text, font, max_width):
    """Word-wrap text to fit within max_width pixels. Returns a list of lines."""
    if not text:
        return [""]

    # First try: does it fit on one line?
    bbox = draw.textbbox((0, 0), text, font=font)
    if (bbox[2] - bbox[0]) <= max_width:
        return [text]

    # Estimate characters per line based on average character width
    avg_char_width = max(1, (bbox[2] - bbox[0]) / max(1, len(text)))
    chars_per_line = max(10, int(max_width / avg_char_width))

    # Use textwrap for clean word breaks
    lines = textwrap.wrap(text, width=chars_per_line)
    if not lines:
        return [text]

    # Refine: if any line still overflows, reduce wrap width
    for _attempt in range(3):
        overflow = False
        for line in lines:
            line_bbox = draw.textbbox((0, 0), line, font=font)
            if (line_bbox[2] - line_bbox[0]) > max_width:
                overflow = True
                break
        if not overflow:
            break
        chars_per_line = max(8, chars_per_line - 4)
        lines = textwrap.wrap(text, width=chars_per_line)

    return lines if lines else [text]


def _draw_centered_text(draw, text, y, img_w, font, stroke_width=2,
                         fill=(255, 255, 255, 255), stroke_fill=(0, 0, 0, 255)):
    """Draw a single line of text centered horizontally with stroke outline."""
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    text_w = bbox[2] - bbox[0]
    x = max(10, (img_w - text_w) // 2)
    draw.text(
        (x, y), text, font=font, fill=fill,
        stroke_width=stroke_width, stroke_fill=stroke_fill,
    )


def make_meme(cv2_bgr_image, caption, bottom_text=None, font_size=None):
    """
    Composites a meme caption at the BOTTOM of the image with a semi-transparent
    black bar. Works correctly on any aspect ratio (portrait, landscape, square,
    ultra-wide).

    cv2_bgr_image: an OpenCV BGR numpy array (e.g. the cartoonified face).
    caption: the main caption text (the generated caption).
    bottom_text: optional tag line (e.g. "HAPPY - SARCASTIC - SARCASM CUE").
    Returns a PIL Image (RGB), ready to .save(...).
    """
    rgb = cv2.cvtColor(cv2_bgr_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).convert("RGBA")
    w, h = img.size

    # Scale font based on the shorter dimension -- works for any aspect ratio
    short_side = min(w, h)
    font_size = font_size or max(18, short_side // 16)
    font = _load_font(font_size)
    tag_font_size = max(12, font_size // 2)
    tag_font = _load_font(tag_font_size)

    # Create transparent overlay for the bar
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Wrap the caption text
    caption_upper = caption.upper()
    padding_x = max(10, w // 30)
    wrapped_lines = _wrap_text(draw, caption_upper, font, w - 2 * padding_x)

    # Calculate bar dimensions
    line_height = font_size + 6
    caption_block_height = len(wrapped_lines) * line_height
    tag_height = (tag_font_size + 10) if bottom_text else 0
    bar_padding_y = max(10, short_side // 40)
    bar_height = caption_block_height + tag_height + 2 * bar_padding_y
    bar_top = h - bar_height

    # Draw semi-transparent black bar (alpha=190 out of 255 ≈ 75% opaque)
    draw.rectangle([(0, bar_top), (w, h)], fill=(0, 0, 0, 190))

    # Draw caption lines (centered, white with black stroke)
    y = bar_top + bar_padding_y
    for line in wrapped_lines:
        _draw_centered_text(draw, line, y, w, font, stroke_width=2)
        y += line_height

    # Draw tag/label (centered, cyan, smaller font)
    if bottom_text:
        tag_upper = bottom_text.upper()
        tag_y = h - tag_height - bar_padding_y // 2
        _draw_centered_text(
            draw, tag_upper, tag_y, w, tag_font,
            stroke_width=1,
            fill=(0, 255, 255, 255),
            stroke_fill=(0, 0, 0, 200),
        )

    # Composite overlay onto image, convert back to RGB for saving
    result = Image.alpha_composite(img, overlay).convert("RGB")
    return result


def save_meme(pil_image, filename):
    """Save the meme image to the output directory."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, filename)
    pil_image.save(path, quality=95)
    return path
