"""
Script to generate the comprehensive, publication-grade Pixora AI Command Reference PDF.

Title: Pixora AI: Command & Control Reference Manual (v2.5.0)
Author: Pixora AI Research & Engineering Team
"""

import os
import sys
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for adding total page count, running header, and footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#0B192C"))

        # Running header on pages > 1
        if self._pageNumber > 1:
            self.drawString(54, 11 * 72 - 36, "PIXORA AI v2.5.0: COMMAND & CONTROL REFERENCE MANUAL")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#707E94"))
            self.drawRightString(8.5 * 72 - 54, 11 * 72 - 36, "HOTKEYS, CLI SCRIPTS & CONFIG REFERENCE")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)

        # Running footer on all pages
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 8.5 * 72 - 54, 45)

        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(54, 32, "Pixora AI v2.5.0 — Operator Field Manual & Reference Guide")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 32, page_str)
        self.restoreState()


def build_pdf(filename="docs/Pixora_AI_Command_Reference.pdf"):
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    C_PRIMARY = colors.HexColor("#0B192C")      # Deep Navy/Slate
    C_ACCENT = colors.HexColor("#008DDA")       # Electric Cyan/Blue
    C_MAGENTA = colors.HexColor("#FF204E")      # Sarcasm Alert Magenta
    C_DARK = colors.HexColor("#1E293B")         # Dark Slate Body Text
    C_BG_CARD = colors.HexColor("#F8FAFC")      # Light Cool Slate
    C_BORDER = colors.HexColor("#CBD5E1")       # Subtle Border
    C_CODE_BG = colors.HexColor("#0F172A")      # Dark Terminal Code Box
    C_CODE_TXT = colors.HexColor("#38BDF8")     # Cyan Code Text

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=C_PRIMARY,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=C_ACCENT,
        spaceAfter=10
    )
    h1_style = ParagraphStyle(
        'Header1',
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=C_PRIMARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'Header2',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=C_ACCENT,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyDark',
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=C_DARK,
        spaceAfter=4
    )
    code_style = ParagraphStyle(
        'CodeSnippet',
        fontName='Courier',
        fontSize=7.5,
        leading=10,
        textColor=C_CODE_TXT
    )
    th_style = ParagraphStyle(
        'TableHeader',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=colors.white
    )
    tb_key_style = ParagraphStyle(
        'TableBodyKey',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=C_MAGENTA
    )
    tb_cmd_style = ParagraphStyle(
        'TableBodyCmd',
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=C_PRIMARY
    )
    tb_desc_style = ParagraphStyle(
        'TableBodyDesc',
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=C_DARK
    )

    story = []

    # Title & Header Banner
    story.append(Paragraph("PIXORA AI v2.5.0: COMMAND & CONTROL REFERENCE MANUAL", title_style))
    story.append(Paragraph("COMPLETE HOTKEY BINDINGS, TERMINAL EXECUTION SCRIPTS & CONFIGURATION MATRIX", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=C_ACCENT, spaceAfter=8))

    # Meta Overview Box
    meta_text = (
        "<b>System Version:</b> Pixora AI v2.5.0 &nbsp;|&nbsp; "
        "<b>Target Platforms:</b> Windows 10/11 (CUDA Enabled), Linux, macOS &nbsp;|&nbsp; "
        "<b>Python Target:</b> 3.13 &nbsp;|&nbsp; "
        "<b>Hardware:</b> NVIDIA RTX (CUDA) / CPU Fallback"
    )
    meta_table = Table([[Paragraph(meta_text, body_style)]], colWidths=[504])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), C_BG_CARD),
        ('BOX', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # SECTION 1: INTERACTIVE KEYBOARD CONTROLS
    story.append(Paragraph("1. Interactive Live Keyboard Controls (OpenCV Window)", h1_style))
    story.append(Paragraph("These hotkeys operate live with sub-millisecond dispatch inside the active camera window.", body_style))

    hotkey_data = [
        [Paragraph("Key", th_style), Paragraph("Command", th_style), Paragraph("Action & Technical Scope", th_style)],
        [Paragraph("SPACE", tb_key_style), Paragraph("Typed Sarcasm Capture", tb_cmd_style), Paragraph("Freezes live video frame; prompts terminal for text; runs multimodal sarcasm fusion (Sfused) & meme generator.", tb_desc_style)],
        [Paragraph("V", tb_key_style), Paragraph("Voice Capture", tb_cmd_style), Paragraph("Captures face while recording 4s audio at 16kHz PCM via local Whisper STT; routes to sarcasm fusion pipeline.", tb_desc_style)],
        [Paragraph("T", tb_key_style), Paragraph("Toggle Cartoon Preview", tb_cmd_style), Paragraph("Toggles live real-time cartoon rendering on all detected faces in the video stream.", tb_desc_style)],
        [Paragraph("M", tb_key_style), Paragraph("Cycle Cartoon Mode", tb_cmd_style), Paragraph("Cycles through 5 modes: avatar_match (Snapchat AR) -> gan -> expression -> cyber_hologram -> opencv.", tb_desc_style)],
        [Paragraph("+ / -", tb_key_style), Paragraph("Avatar Head Scale", tb_cmd_style), Paragraph("Increases / decreases 3D AR avatar head scale in real time (0.5x to 2.5x, step 0.05x).", tb_desc_style)],
        [Paragraph("P", tb_key_style), Paragraph("Feature Passthrough", tb_cmd_style), Paragraph("Blends real user eyes and mouth onto cartoon avatar for a hybrid Snapchat filter aesthetic.", tb_desc_style)],
        [Paragraph("R", tb_key_style), Paragraph("Record Looping GIF", tb_cmd_style), Paragraph("Records a 3-second 30 FPS circular buffer (90 frames) and exports an animated meme GIF to outputs/.", tb_desc_style)],
        [Paragraph("G", tb_key_style), Paragraph("GenAI Scene Meme", tb_cmd_style), Paragraph("Executes zero-shot Vision-Language Model (VLM) scene context analysis + LLM meme synthesis.", tb_desc_style)],
        [Paragraph("A", tb_key_style), Paragraph("AI Comedy Roast", tb_cmd_style), Paragraph("Analyzes live biometrics, micro-expressions, and posture to generate a savage AI comedy roast.", tb_desc_style)],
        [Paragraph("S", tb_key_style), Paragraph("Voice Persona (TTS)", tb_cmd_style), Paragraph("Toggles spoken speech synthesis (SAPI5/pyttsx3) for meme punchlines and AI roasts.", tb_desc_style)],
        [Paragraph("F", tb_key_style), Paragraph("Sci-Fi Sound FX", tb_cmd_style), Paragraph("Toggles cybernetic interface sound cues (target acquisition chirps, shutter clicks, alert pings).", tb_desc_style)],
        [Paragraph("C", tb_key_style), Paragraph("Cycle Caption Style", tb_cmd_style), Paragraph("Cycles comedic tone styles: Funny, Sarcastic, Witty, Ironic, Motivational.", tb_desc_style)],
        [Paragraph("D", tb_key_style), Paragraph("Toggle Debug HUD", tb_cmd_style), Paragraph("Toggles green 478-mesh landmark points, iris gaze reticles, and live FACS cue strength bars.", tb_desc_style)],
        [Paragraph("Q / Esc", tb_key_style), Paragraph("Quit Application", tb_cmd_style), Paragraph("Safely releases webcam, GPU tensors, audio threads, and terminates cleanly.", tb_desc_style)],
    ]
    hotkey_table = Table(hotkey_data, colWidths=[50, 110, 344])
    hotkey_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, C_BG_CARD]),
        ('PADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(hotkey_table)
    story.append(Spacer(1, 8))

    # SECTION 2: RUNTIME & APPLICATION LAUNCH COMMANDS
    story.append(Paragraph("2. Application Launch & Web Perception Hub Commands", h1_style))
    
    app_cli_data = [
        [Paragraph("Target Component", th_style), Paragraph("CLI Execution Command", th_style), Paragraph("Description & Port", th_style)],
        [
            Paragraph("Desktop Cyberpunk AR", tb_cmd_style),
            Paragraph("venv\\Scripts\\python.exe main.py", code_style),
            Paragraph("Launches primary 30+ FPS OpenCV desktop app with biometrics and Procrustes AR avatar.", tb_desc_style)
        ],
        [
            Paragraph("Flask Web Perception Hub", tb_cmd_style),
            Paragraph("venv\\Scripts\\python.exe web_app.py", code_style),
            Paragraph("Starts browser streaming server at http://localhost:5000 with remote controls & graphs.", tb_desc_style)
        ],
    ]
    app_cli_table = Table(app_cli_data, colWidths=[120, 190, 194])
    app_cli_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, C_BG_CARD]),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(app_cli_table)
    story.append(Spacer(1, 8))

    # SECTION 3: ENVIRONMENT & CUDA INSTALLATION
    story.append(Paragraph("3. Environment Setup & CUDA Commands", h1_style))

    env_data = [
        [Paragraph("Task", th_style), Paragraph("PowerShell / Bash Command", th_style)],
        [Paragraph("Create venv (Python 3.13)", tb_cmd_style), Paragraph("py -3.13 -m venv venv && venv\\Scripts\\Activate.ps1", code_style)],
        [Paragraph("Install PyTorch CUDA", tb_cmd_style), Paragraph("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124", code_style)],
        [Paragraph("Verify CUDA GPU", tb_cmd_style), Paragraph("python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\"", code_style)],
        [Paragraph("Install Core Requirements", tb_cmd_style), Paragraph("pip install -r requirements.txt", code_style)],
        [Paragraph("Install Advanced Requirements", tb_cmd_style), Paragraph("pip install -r requirements-advanced.txt", code_style)],
    ]
    env_table = Table(env_data, colWidths=[130, 374])
    env_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, C_BG_CARD]),
        ('PADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(env_table)
    story.append(Spacer(1, 8))

    story.append(PageBreak())

    # SECTION 4: NEURAL TRAINING & INDEXING SCRIPTS
    story.append(Paragraph("4. Neural Training & Avatar Indexing Commands", h1_style))
    story.append(Paragraph("All models can be retrained locally on GPU or in Google Colab (Pixora_Online_Colab.ipynb).", body_style))

    train_data = [
        [Paragraph("Pipeline Component", th_style), Paragraph("Execution Script & Parameters", th_style), Paragraph("Benchmark & Hardware", th_style)],
        [
            Paragraph("30k Avatar Deep Index", tb_cmd_style),
            Paragraph("python training/build_cartoon_avatar_index.py", code_style),
            Paragraph("MobileNetV3 960-D + LAB color indexing across 30k Google Cartoon Set images.", tb_desc_style)
        ],
        [
            Paragraph("Baseline Sarcasm Model", tb_cmd_style),
            Paragraph("python training/train_baseline_local.py", code_style),
            Paragraph("85.8% Acc / 0.851 F1 on 28.6k headlines (Intel CPU ~1.2s).", tb_desc_style)
        ],
        [
            Paragraph("Advanced DeBERTa-v3 Sarcasm", tb_cmd_style),
            Paragraph("python training/train_advanced_local.py --batch-size 8", code_style),
            Paragraph("91.4% Acc on held-out test split (RTX 4060 GPU ~4.5 min).", tb_desc_style)
        ],
        [
            Paragraph("MUStARD Dialogue Adaptation", tb_cmd_style),
            Paragraph("python training/finetune_mustard_local.py", code_style),
            Paragraph("+14.2% in-domain conversational sarcasm accuracy gain.", tb_desc_style)
        ],
        [
            Paragraph("Custom 4-Block Emotion CNN", tb_cmd_style),
            Paragraph("python training/train_emotion_local.py --epochs 30", code_style),
            Paragraph("68.7% test accuracy on 35.8k FER-2013 facial expression crops.", tb_desc_style)
        ],
    ]
    train_table = Table(train_data, colWidths=[120, 204, 180])
    train_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, C_BG_CARD]),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(train_table)
    story.append(Spacer(1, 8))

    # SECTION 5: DOCUMENTATION & ARTIFACT GENERATORS
    story.append(Paragraph("5. Documentation, Report & Diagram Generators", h1_style))

    gen_data = [
        [Paragraph("Artifact / Report", th_style), Paragraph("Generator Script", th_style), Paragraph("Output File Path", th_style)],
        [
            Paragraph("Project Overview PDF", tb_cmd_style),
            Paragraph("python generate_overview_pdf.py", code_style),
            Paragraph("docs/Pixora_AI_Project_Overview.pdf", tb_desc_style)
        ],
        [
            Paragraph("Command Reference PDF", tb_cmd_style),
            Paragraph("python generate_command_reference_pdf.py", code_style),
            Paragraph("docs/Pixora_AI_Command_Reference.pdf", tb_desc_style)
        ],
        [
            Paragraph("Draw.io Multi-Tab Architecture", tb_cmd_style),
            Paragraph("python generate_drawio_diagram.py", code_style),
            Paragraph("pixora_architecture.drawio (3 Tabs)", tb_desc_style)
        ],
    ]
    gen_table = Table(gen_data, colWidths=[130, 184, 190])
    gen_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, C_BG_CARD]),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(gen_table)
    story.append(Spacer(1, 8))

    # SECTION 6: CONFIGURATION PARAMETERS (config.py)
    story.append(Paragraph("6. Key Configuration Parameters (config.py)", h1_style))

    cfg_data = [
        [Paragraph("Setting Parameter", th_style), Paragraph("Default", th_style), Paragraph("Description & Options", th_style)],
        [Paragraph("CARTOONIFY_MODE", tb_cmd_style), Paragraph('"avatar_match"', code_style), Paragraph('Active filter: "avatar_match", "gan", "expression", "cyber_hologram", "opencv"', tb_desc_style)],
        [Paragraph("GAN_STYLE", tb_cmd_style), Paragraph('"portrait_v2"', code_style), Paragraph('AnimeGAN style: "portrait_v2", "portrait_v1", "celeba_distill", "paprika"', tb_desc_style)],
        [Paragraph("CARTOON_AVATAR_SCALE", tb_cmd_style), Paragraph('1.15', code_style), Paragraph('Head scale factor for Snapchat Procrustes AR avatar alignment.', tb_desc_style)],
        [Paragraph("CARTOON_PASSTHROUGH_EYES", tb_cmd_style), Paragraph('False', code_style), Paragraph('Blends real user eyes and mouth onto cartoon avatar.', tb_desc_style)],
        [Paragraph("RPPG_HEART_RATE_ENABLED", tb_cmd_style), Paragraph('True', code_style), Paragraph('Enables contactless optical capillary pulse & stress estimation.', tb_desc_style)],
        [Paragraph("EYE_GAZE_TRACKING_ENABLED", tb_cmd_style), Paragraph('True', code_style), Paragraph('Enables 3D eye gaze quadrant and laser reticle tracking.', tb_desc_style)],
        [Paragraph("CAPTION_GENERATION_BACKEND", tb_cmd_style), Paragraph('"local"', code_style), Paragraph('Caption engine: "local" (Qwen2.5-1.5B), "api" (Groq/Gemini), "template"', tb_desc_style)],
        [Paragraph("AUDIO_VOICE_ENABLED", tb_cmd_style), Paragraph('False', code_style), Paragraph('Spoken TTS voice persona for meme punchlines and AI roasts.', tb_desc_style)],
    ]
    cfg_table = Table(cfg_data, colWidths=[140, 75, 289])
    cfg_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, C_BG_CARD]),
        ('PADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(cfg_table)
    story.append(Spacer(1, 8))

    # SECTION 7: DIAGNOSTIC & TROUBLESHOOTING COMMANDS
    story.append(Paragraph("7. Diagnostic & Troubleshooting Commands", h1_style))

    diag_data = [
        [Paragraph("Issue / Symptom", th_style), Paragraph("Diagnostic & Resolution Command", th_style)],
        [Paragraph("No video window appearing", tb_cmd_style), Paragraph("pip uninstall opencv-python-headless -y && pip install opencv-python", code_style)],
        [Paragraph("DeepFace / tf-keras error", tb_cmd_style), Paragraph("pip install tf-keras", code_style)],
        [Paragraph("CUDA Out of Memory in Training", tb_cmd_style), Paragraph("python training/train_advanced_local.py --batch-size 4 --grad-accum 8", code_style)],
        [Paragraph("Kill stuck python processes", tb_cmd_style), Paragraph("Stop-Process -Name python -Force", code_style)],
    ]
    diag_table = Table(diag_data, colWidths=[140, 364])
    diag_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C_PRIMARY),
        ('GRID', (0, 0), (-1, -1), 0.5, C_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, C_BG_CARD]),
        ('PADDING', (0, 0), (-1, -1), 3.5),
    ]))
    story.append(diag_table)

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[PDF] Successfully generated Pixora AI Command Reference PDF: {filename}")


if __name__ == "__main__":
    out_path = "docs/Pixora_AI_Command_Reference.pdf"
    build_pdf(out_path)
    # Also generate root copy for easy accessibility
    build_pdf("Pixora_AI_Command_Reference.pdf")
