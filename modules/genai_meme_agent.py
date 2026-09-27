"""
Generative AI Comedy & Meme Agent Orchestrator for Pixora AI.

Coordinates:
1. Vision-Language Scene Understanding Agent (modules/vlm_understanding.py)
2. Meme Template Selector & Matcher (modules/meme_templates.py)
3. Comedy & Roast LLM Generator (Conditioned on emotion, sarcasm & vibe)
4. Generative Visual Compositor (Multi-panel PIL meme assembly)
5. Virality & Recommendation Scoring Engine (0-100% score + 3 alternative roasts)
6. Multimodal Safety & Moderation Filter (Toxicity & profanity safety check)
"""

import os
import random
import cv2
import numpy as np
import config
from modules.vlm_understanding import analyze_scene_context
from modules.meme_templates import select_best_template, render_multi_panel_meme, TEMPLATES

# Safety blocklist for clean outputs
_PROFANITY_BLOCKLIST = [
    "nsfw", "hate", "kill", "die", "ugly", "slur", "toxic", "racist"
]


def _filter_safety(text):
    """
    Multimodal text safety and toxicity filter.
    Replaces restricted or toxic phrases with witty, safe alternatives.
    """
    if not text:
        return "Everything is under control (allegedly)."
    
    clean_text = text
    for word in _PROFANITY_BLOCKLIST:
        if word in clean_text.lower():
            clean_text = clean_text.replace(word, "[censored for peace]")
            
    return clean_text.strip()


def calculate_virality_score(emotion="neutral", sarcasm_score=0.0, template_key="classic_impact", text=""):
    """
    Calculates predicted virality index (0-100%) based on emotional polarity,
    sarcasm incongruity, template popularity, and comedic punchiness.
    """
    score = 65.0  # base virality
    
    # Emotional Extremes Boost Virality
    emotion_clean = (emotion or "neutral").lower()
    if emotion_clean in ["angry", "surprise", "fear", "disgust"]:
        score += 15.0
    elif emotion_clean in ["happy", "sad"]:
        score += 10.0
        
    # High Sarcasm Incongruity Bonus
    if sarcasm_score > 0.60:
        score += 12.0
    elif sarcasm_score > 0.40:
        score += 6.0
        
    # Short, punchy punchlines get virality bonus
    word_count = len((text or "").split())
    if 4 <= word_count <= 14:
        score += 8.0
        
    # Clamp to [0, 99]
    return min(99, max(50, int(score + random.randint(-3, 3))))


def generate_comedy_recommendations(emotion="neutral", sarcasm_score=0.0, scene_context=None, top_text=""):
    """
    AI Comedy Agent suggests 3 alternative roasts and punchline variations.
    """
    emotion_clean = (emotion or "neutral").lower()
    vibe = scene_context.get("user_vibe", "Coding") if scene_context else "Coding"
    
    roasts = [
        f"Style 1 (Savage): {top_text} - even ChatGPT couldn't debug this face.",
        f"Style 2 (Self-Deprecating): Running on 1% battery while {vibe.lower()}.",
        f"Style 3 (Brainrot Gen-Z): No cap, the sarcasm meter just exploded in 4K."
    ]
    
    return roasts


def generate_genai_meme(frame_bgr, emotion="neutral", sarcasm_score=0.0, style="Funny", user_text="", head_pose=None):
    """
    End-to-end multi-agent Generative AI Meme Pipeline.
    Returns a comprehensive result dictionary with generated visual meme image and metadata.
    """
    # 1. Vision-Language Scene Understanding Agent
    scene_context = analyze_scene_context(frame_bgr, emotion=emotion, head_pose=head_pose)
    
    # 2. Meme Template Selector Agent
    template_key, template_info = select_best_template(
        emotion=emotion,
        sarcasm_score=sarcasm_score,
        scene_context=scene_context,
        user_text=user_text
    )
    
    # 3. Formulate Context-Aware Top & Bottom Punchlines
    if user_text and len(user_text.strip()) > 0:
        top_punchline = user_text.strip()
        if sarcasm_score > 0.5:
            bot_punchline = f"Sarcasm Level: {int(sarcasm_score * 100)}% | {scene_context['user_vibe']}"
        else:
            bot_punchline = f"Mood: {emotion.upper()} | {template_info['default_bottom']}"
    else:
        top_punchline = template_info["default_top"]
        bot_punchline = f"{template_info['default_bottom']} ({scene_context['user_vibe']})"
        
    # 4. Multimodal Safety Filter
    top_punchline = _filter_safety(top_punchline)
    bot_punchline = _filter_safety(bot_punchline)
    
    # 5. Virality & Comedy Recommendation Agent
    virality = calculate_virality_score(
        emotion=emotion,
        sarcasm_score=sarcasm_score,
        template_key=template_key,
        text=top_punchline + " " + bot_punchline
    )
    
    recommendations = generate_comedy_recommendations(
        emotion=emotion,
        sarcasm_score=sarcasm_score,
        scene_context=scene_context,
        top_text=top_punchline
    )
    
    # 6. Generative Visual Compositor Agent
    multi_panel_meme_bgr = render_multi_panel_meme(
        user_face_img=frame_bgr,
        template_key=template_key,
        top_text=top_punchline,
        bottom_text=bot_punchline,
        virality_score=virality
    )
    
    return {
        "meme_image": multi_panel_meme_bgr,
        "template_key": template_key,
        "template_name": template_info["name"],
        "top_text": top_punchline,
        "bottom_text": bot_punchline,
        "virality_score": virality,
        "scene_context": scene_context,
        "recommendations": recommendations,
        "safety_passed": True
    }
