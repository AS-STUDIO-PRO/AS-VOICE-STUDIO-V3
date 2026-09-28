<div align="center">

# 🎙️ AS VOICE STUDIO V3

**Professional multilingual voice cloning, design and production — in one clean workspace.**

[

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

](https://colab.research.google.com/github/AS-STUDIO-PRO/AS-VOICE-STUDIO-V3/blob/main/AS_VOICE_STUDIO_V3.ipynb)


![Python](https://img.shields.io/badge/Python-3.10%2B-blue)




![Interface](https://img.shields.io/badge/Interface-Gradio%205-6366f1)




![Languages](https://img.shields.io/badge/Languages-600%2B-success)




![GPU](https://img.shields.io/badge/GPU-Colab%20T4%20ready-orange)



</div>

---

## ✨ Features

| Tab | What it does |
|---|---|
| **Voice Clone** | Clone a voice from a 3–10 second clip. Reference transcript is optional (auto-transcribed). Save the voice for later. |
| **Voice Design** | Build a voice from scratch: gender, age, pitch, style, accent, plus one-click presets (Narrator, News Anchor, Storyteller, Presenter, Soft Whisper). |
| **Auto Voice** | Let the studio pick a voice automatically. |
| **Expressions** | Insert non-verbal tags such as `[laughter]` and `[sigh]` directly into your script. Supports phoneme-based pronunciation fixes. |
| **Voice Library** | Reuse saved voices across sessions without re-uploading the reference. |
| **Batch Export** | One line per file. Renders every line and downloads everything as a single ZIP. |
| **Guide** | Live system status, tips and responsible-use notes. |

Also included: automatic sentence-level splitting for long scripts, adjustable quality and speaking speed, 24 kHz WAV output, and a clean light professional interface that works on desktop and mobile.

## 🚀 Quick start (Google Colab)

1. Click **Open in Colab** above.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Run **Cell 1** (installs everything).
4. Run **Cell 2** — a public link appears. Open it and start creating.

## 💻 Run locally

```bash
git clone https://github.com/AS-STUDIO-PRO/AS-VOICE-STUDIO-V3.git
cd AS-VOICE-STUDIO-V3
pip install -r requirements.txt
python app.py            # opens on http://localhost:7860
