"""
Iconic Meme Template Library & Semantic Selection Engine for Pixora AI.

Provides 12 classic and modern meme layout structures:
1. "drake"           -- Drake Hotline Bling (Top: Dislike / Bottom: Like)
2. "this_is_fine"    -- Dog in Burning Room (Everything is totally fine)
3. "two_buttons"     -- Daily struggle between two panic choices
4. "distracted_bf"   -- Wandering eye looking away from duty
5. "expanding_brain" -- 3-Stage escalating genius / madness
6. "doge_cheems"     -- Buff Doge vs Crying Cheems (Expectation vs Reality)
7. "spongebob_mock"  -- Mocking SpongeBob text formatting
8. "woman_yelling"   -- Woman yelling at innocent confused cat
9. "gigachad"        -- Unbothered ultra-confident stance
10. "panik_kalm"     -- 3-Panel emotional rollercoaster
11. "clown_makeup"   -- 4-Stage escalating delusion
12. "classic_impact" -- Standard high-contrast top/bottom header
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import config
from modules.meme import _load_font, _wrap_text, _draw_centered_text

TEMPLATES = {
    "drake": {
        "name": "Drake Hotline Bling",
        "description": "Rejecting the logical option, embracing the chaotic one",
        "panels": 2,
        "tags": ["choice", "rejection", "preference", "coding", "sarcastic"],
        "default_top": "Writing clean, tested code with documentation",
        "default_bottom": "Pushing straight to main with commit message 'fix stuff'"
    },
    "this_is_fine": {
        "name": "This Is Fine",
        "description": "Smiling calmly while the surrounding architecture burns down",
        "panels": 1,
        "tags": ["panic", "fire", "deadline", "calm", "denial", "angry", "fear"],
        "default_top": "EVERYTHING IS GOING ACCORDING TO PLAN",
        "default_bottom": "(47 errors, 12 warnings, 0 tests passing)"
    },
    "two_buttons": {
        "name": "Two Buttons Sweat",
        "description": "Agonizing choice between two equally risky alternatives",
        "panels": 2,
        "tags": ["dilemma", "sweating", "anxiety", "fear", "surprise"],
        "default_top": "Button A: Sleep 8 Hours | Button B: Fix 1 More Bug",
        "default_bottom": "*Sweating profusely at 3:45 AM*"
    },
    "distracted_bf": {
        "name": "Distracted Developer",
        "description": "Ignoring existing stable responsibilities for new shiny frameworks",
        "panels": 2,
        "tags": ["temptation", "distraction", "abandonment", "happy", "witty"],
        "default_top": "My pending Jira tickets",
        "default_bottom": "Rewriting the entire backend in Rust"
    },
    "expanding_brain": {
        "name": "Expanding Brain Galaxy",
        "description": "3-Stage evolution from mundane to ascended cosmic thinking",
        "panels": 3,
        "tags": ["evolution", "galaxy", "genius", "sarcastic", "witty"],
        "default_top": "Level 1: Print statements | Level 2: Debugger",
        "default_bottom": "Level 3: Staring at the code until it confesses"
    },
    "doge_cheems": {
        "name": "Buff Doge vs Cheems",
        "description": "Glorious past expectation vs crying pathetic reality",
        "panels": 2,
        "tags": ["comparison", "strong", "weak", "sad", "ironic"],
        "default_top": "Senior Dev in 2004: Wrote OS in assembly",
        "default_bottom": "Me today: Needs StackOverflow to center a div"
    },
    "spongebob_mock": {
        "name": "Mocking SpongeBob",
        "description": "Repeating the opponent\'s absurd sentence in alternating caps",
        "panels": 1,
        "tags": ["mocking", "sarcastic", "retort", "disgust", "angry"],
        "default_top": "'iT wOrKs On My MaChInE'",
        "default_bottom": "tHeN wE wIlL sHiP yOuR mAcHiNe To ClOuD"
    },
    "woman_yelling": {
        "name": "Woman Yelling at Cat",
        "description": "Furious accusation met with utter oblivious innocence",
        "panels": 2,
        "tags": ["conflict", "innocent", "confusion", "angry", "surprise"],
        "default_top": "YOU BROKE THE PRODUCTION DATABASE!",
        "default_bottom": "Me who just ran 'SELECT * FROM users;'"
    },
    "gigachad": {
        "name": "Gigachad Confidence",
        "description": "Unapologetic, unbothered ultra-confidence in questionable choices",
        "panels": 1,
        "tags": ["chad", "confident", "unbothered", "happy", "motivational"],
        "default_top": "Yes, I test in production.",
        "default_bottom": "How could you tell?"
    },
    "panik_kalm": {
        "name": "Panik Kalm Panik",
        "description": "3-Stage roller coaster of sudden disaster, false relief, and true doom",
        "panels": 3,
        "tags": ["rollercoaster", "panic", "relief", "fear", "surprise"],
        "default_top": "PANIK: Server down! -> KALM: It was just cache...",
        "default_bottom": "PANIK: Production database was dropped!"
    },
    "clown_makeup": {
        "name": "Clown Makeup Evolution",
        "description": "Step-by-step descending into complete comedic delusion",
        "panels": 2,
        "tags": ["clown", "delusion", "fool", "sad", "ironic"],
        "default_top": "'I will just do a quick 5-minute refactor'",
        "default_bottom": "*6 hours later rewriting entire architecture*"
    },
    "classic_impact": {
        "name": "Classic Impact Meme",
        "description": "Bold top and bottom punchline across the full frame",
        "panels": 1,
        "tags": ["classic", "general", "impact", "all"],
        "default_top": "ONE DOES NOT SIMPLY",
        "default_bottom": "EXPLAIN THIS TO THE PRODUCT MANAGER"
    }
}


def select_best_template(emotion="neutral", sarcasm_score=0.0, scene_context=None, user_text=""):
    """
    Semantic cosine & tag matching engine to select the best template for the moment.
    """
    forced = getattr(config, "MEME_TEMPLATE_SELECTION", "auto").lower()
    if forced in TEMPLATES:
        return forced, TEMPLATES[forced]

    emotion_clean = (emotion or "neutral").lower()
    text_lower = (user_text or "").lower()
    vibe_lower = (scene_context.get("user_vibe", "") if scene_context else "").lower()

    scores = {k: 0.0 for k in TEMPLATES}

    # Match based on emotion
    for k, t in TEMPLATES.items():
        if emotion_clean in t["tags"]:
            scores[k] += 2.5
        if "all" in t["tags"]:
            scores[k] += 1.0

    # Match based on sarcasm score
    if sarcasm_score > 0.5:
        scores["drake"] += 1.5
        scores["spongebob_mock"] += 2.0
        scores["this_is_fine"] += 1.5
        scores["clown_makeup"] += 1.5
    else:
        scores["gigachad"] += 1.0
        scores["classic_impact"] += 1.0

    # Match based on user keywords
    for k, t in TEMPLATES.items():
        for tag in t["tags"]:
            if tag in text_lower or tag in vibe_lower:
                scores[k] += 2.0

    best_key = max(scores, key=scores.get)
    return best_key, TEMPLATES[best_key]


def render_multi_panel_meme(user_face_img, template_key, top_text, bottom_text, virality_score=85):
    """
    Composites user face image into multi-panel meme layout with high-contrast borders,
    Impact headers, and virality badges. Returns BGR numpy array.
    """
    if template_key not in TEMPLATES:
        template_key = "classic_impact"

    t = TEMPLATES[template_key]
    h, w = user_face_img.shape[:2]

    # Standardize output canvas to 800x600 for sharp readability
    out_w, out_h = 800, 600
    canvas = Image.new("RGBA", (out_w, out_h), (11, 15, 25, 255))
    draw = ImageDraw.Draw(canvas)

    # Convert user face crop to PIL Image
    rgb_face = cv2.cvtColor(user_face_img, cv2.COLOR_BGR2RGB)
    pil_face = Image.fromarray(rgb_face)

    font_main = _load_font(28)
    font_sub = _load_font(18)
    font_badge = _load_font(15)

    if t["panels"] == 2:
        # Two-panel horizontal / vertical layout
        panel_h = (out_h - 120) // 2
        
        # Panel 1 (Top)
        p1_face = pil_face.resize((260, panel_h))
        canvas.paste(p1_face, (20, 60))
        # Draw red border for dislike/option 1
        draw.rectangle([18, 58, 282, 60 + panel_h + 2], outline=(239, 68, 68, 255), width=3)
        # Top text box
        draw.rounded_rectangle([300, 60, out_w - 20, 60 + panel_h], radius=8, fill=(30, 41, 59, 230), outline=(239, 68, 68, 200), width=2)
        top_lines = _wrap_text(draw, top_text or t["default_top"], font_main, out_w - 340)
        y_start = 60 + (panel_h - len(top_lines) * 34) // 2
        for line in top_lines:
            draw.text((320, y_start), line, font=font_main, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
            y_start += 34

        # Panel 2 (Bottom)
        p2_y = 70 + panel_h
        p2_face = pil_face.resize((260, panel_h))
        canvas.paste(p2_face, (20, p2_y))
        # Draw green border for like/option 2
        draw.rectangle([18, p2_y - 2, 282, p2_y + panel_h + 2], outline=(16, 185, 129, 255), width=3)
        # Bottom text box
        draw.rounded_rectangle([300, p2_y, out_w - 20, p2_y + panel_h], radius=8, fill=(30, 41, 59, 230), outline=(16, 185, 129, 200), width=2)
        bot_lines = _wrap_text(draw, bottom_text or t["default_bottom"], font_main, out_w - 340)
        y_start = p2_y + (panel_h - len(bot_lines) * 34) // 2
        for line in bot_lines:
            draw.text((320, y_start), line, font=font_main, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
            y_start += 34

    else:
        # Single panel / full frame impact layout
        scaled_face = pil_face.resize((out_w - 40, out_h - 120))
        canvas.paste(scaled_face, (20, 60))
        draw.rectangle([18, 58, out_w - 18, out_h - 58], outline=(0, 229, 255, 255), width=3)

        # Top Text Bar
        top_txt = top_text or t["default_top"]
        if top_txt:
            draw.rounded_rectangle([30, 70, out_w - 30, 130], radius=8, fill=(0, 0, 0, 200), outline=(0, 229, 255, 180), width=2)
            _draw_centered_text(draw, top_txt, 82, out_w, font_main, stroke_width=2)

        # Bottom Text Bar
        bot_txt = bottom_text or t["default_bottom"]
        if bot_txt:
            draw.rounded_rectangle([30, out_h - 130, out_w - 30, out_h - 70], radius=8, fill=(0, 0, 0, 200), outline=(255, 32, 78, 180), width=2)
            _draw_centered_text(draw, bot_txt, out_h - 118, out_w, font_main, stroke_width=2)

    # Top Header Banner
    draw.rectangle([0, 0, out_w, 45], fill=(15, 23, 42, 255))
    draw.text((20, 12), f"🔥 PIXORA AI MEME: {t['name'].upper()}", font=font_sub, fill=(0, 229, 255, 255))
    
    # Virality Badge
    v_color = (16, 185, 129, 255) if virality_score >= 80 else (245, 158, 11, 255)
    draw.rounded_rectangle([out_w - 210, 8, out_w - 20, 38], radius=6, fill=(30, 41, 59, 255), outline=v_color, width=2)
    draw.text((out_w - 200, 14), f"⚡ VIRALITY: {virality_score}%", font=font_badge, fill=v_color)

    # Bottom Metadata Watermark
    draw.rectangle([0, out_h - 35, out_w, out_h], fill=(15, 23, 42, 255))
    draw.text((20, out_h - 26), "PIXORA AI GENERATIVE AGENT | MULTIMODAL AFFECT ENGINE", font=font_badge, fill=(148, 163, 184, 255))

    out_bgr = cv2.cvtColor(np.array(canvas.convert("RGB")), cv2.COLOR_RGB2BGR)
    return out_bgr
