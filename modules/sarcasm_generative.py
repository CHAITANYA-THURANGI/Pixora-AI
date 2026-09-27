"""
Generative-AI sarcasm verification -- a SECOND, independent way of judging
sarcasm, deliberately NOT based on the trained/dataset-based classifier in
modules/sarcasm_text.py. Where that model learned from 28,619 labeled
headlines (and optionally MUStARD), this module asks a general-purpose
instruction-tuned language model to judge sarcasm zero-shot, from
reasoning alone, the same local model already used for captions
(modules/captions.py) -- no fine-tuning, no training data, no dataset
lookup of any kind.

WHY BOTH EXIST, RATHER THAN REPLACING ONE WITH THE OTHER: a model trained
on labeled data gives you something a zero-shot prompt cannot -- a
reproducible accuracy number on a held-out test set (see
models/sarcasm_baseline/metrics.json and training_metrics.json). A
zero-shot generative judgment gives you something the trained model
cannot -- a plain-language REASON for the verdict, and a check that
doesn't inherit any bias specific to headline or TV-dialogue training
data. Showing both, side by side, is more defensible than either alone,
and it's what modules/sarcasm_fusion.py's voice-input path does.

Used specifically for the voice/speech-input path (main.py's V key), per
project requirements -- typed input still primarily uses the trained
classifier, though this function works for any text.
"""

import json

import config

_llm = None
_llm_tokenizer = None
_llm_load_failed = False

VERIFY_SYSTEM_PROMPT = (
    "You are a careful sarcasm-detection expert. Given a sentence, decide "
    "whether it is sarcastic. Respond with ONLY a compact JSON object, no "
    "other text, in exactly this form: "
    '{"sarcastic": true or false, "confidence": a number from 0 to 1, '
    '"reason": a short one-sentence explanation}.'
)


def _load_local_llm():
    """Reuses the exact same local model/loader as modules/captions.py so
    the model is only ever loaded into GPU memory once, not twice."""
    global _llm, _llm_tokenizer, _llm_load_failed
    if _llm is not None or _llm_load_failed:
        return _llm, _llm_tokenizer

    try:
        from modules import captions as _captions_mod
        model, tokenizer = _captions_mod._load_local_llm()
        if model is None:
            _llm_load_failed = True
            return None, None
        _llm, _llm_tokenizer = model, tokenizer
        return _llm, _llm_tokenizer
    except Exception as e:
        print(f"[sarcasm_generative] Could not load local model ({e}).")
        _llm_load_failed = True
        return None, None


def _parse_verdict(raw_text, fallback_sarcastic):
    """Best-effort JSON parse with a safe fallback -- a live demo should
    never crash because the model's output wasn't perfectly formed JSON."""
    try:
        start = raw_text.index("{")
        end = raw_text.rindex("}") + 1
        parsed = json.loads(raw_text[start:end])
        return {
            "sarcastic": bool(parsed.get("sarcastic", fallback_sarcastic)),
            "confidence": float(parsed.get("confidence", 0.5)),
            "reason": str(parsed.get("reason", "")).strip(),
            "raw_ok": True,
        }
    except Exception:
        # Fall back to a simple keyword check on the raw text rather than
        # giving up entirely -- "false" is checked first since our prompt
        # always asks for the word "false" specifically for a non-sarcastic
        # verdict, making it the more reliable single keyword to key off.
        lowered = raw_text.lower()
        if "false" in lowered:
            sarcastic = False
        elif "true" in lowered or "sarcastic" in lowered:
            sarcastic = True
        else:
            sarcastic = fallback_sarcastic
        return {
            "sarcastic": sarcastic,
            "confidence": 0.5,
            "reason": "(model reply wasn't clean JSON -- parsed heuristically)",
            "raw_ok": False,
        }


def verify_sarcasm_generative(text: str, fallback_sarcastic: bool = False) -> dict:
    """
    Zero-shot sarcasm judgment from the local generative model. Returns a
    dict with sarcastic (bool), confidence (0-1), reason (str), and
    available (bool) -- available=False means the model wasn't loadable
    (e.g. requirements-advanced.txt missing) and every other field is a
    safe fallback, never a crash.
    """
    if not text or not text.strip():
        return {"sarcastic": False, "confidence": 0.0, "reason": "(no text)", "available": False}

    model, tokenizer = _load_local_llm()
    if model is None:
        return {"sarcastic": fallback_sarcastic, "confidence": 0.0,
                "reason": "(generative model unavailable -- see requirements-advanced.txt)",
                "available": False}

    import torch

    messages = [
        {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
        {"role": "user", "content": f'Sentence: "{text}"'},
    ]
    try:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=80,
                do_sample=False,  # deterministic for a classification-style judgment, unlike caption generation
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        raw = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        result = _parse_verdict(raw, fallback_sarcastic)
        result["available"] = True
        return result
    except Exception as e:
        print(f"[sarcasm_generative] Generation failed ({e}); using fallback.")
        return {"sarcastic": fallback_sarcastic, "confidence": 0.0,
                "reason": f"(generation error: {e})", "available": False}
