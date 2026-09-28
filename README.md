<div align="center">

# 🎙️ AS VOICE STUDIO V3

**Professional multilingual voice cloning, design and production — in one clean workspace.**

<a href="https://colab.research.google.com/github/AS-STUDIO-PRO/AS-VOICE-STUDIO-V3/blob/main/AS_VOICE_STUDIO_V3.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

<img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue"> <img alt="Interface" src="https://img.shields.io/badge/Interface-Gradio%205-6366f1"> <img alt="Languages" src="https://img.shields.io/badge/Languages-600%2B-success"> <img alt="GPU" src="https://img.shields.io/badge/GPU-Colab%20T4%20ready-orange">

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

Also included: automatic sentence-level splitting for long scripts, adjustable quality and speaking speed, 24 kHz WAV output, a live activity log in the Colab cell, and a clean light interface that works on desktop and mobile.

## 🚀 Quick start (Google Colab)

1. Click the **Open in Colab** button above.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Run **Cell 1** (installs everything).
4. Run **Cell 2** — the studio starts and a public link appears in the output. Open it and start creating.
5. Keep Cell 2 running while you work. The live activity log shows every request.

## 💻 Run locally

~~~bash
git clone https://github.com/AS-STUDIO-PRO/AS-VOICE-STUDIO-V3.git
cd AS-VOICE-STUDIO-V3
pip install -r requirements.txt
python app.py            # opens on http://localhost:7860
~~~

A CUDA GPU is strongly recommended. Set `AS_DEMO_MODE=1` to test the interface without a GPU or model download.

## 🎯 Tips for best results

- Use a **clean 3–10 second** reference clip with no background noise or music.
- Keep the **reference and target language the same** to avoid an accent carry-over.
- Quality: **16** = fast, **32** = balanced, **48+** = studio.
- Write numbers as words (`123` → `one hundred twenty-three`) for correct pronunciation.
- Long scripts are split at sentence boundaries automatically and joined with a short pause.

## 🗂️ Project structure

~~~text
├── app.py                      # Interface and workflow logic
├── engine.py                   # Voice engine wrapper
├── AS_VOICE_STUDIO_V3.ipynb    # One-click Colab launcher
├── requirements.txt
├── NOTICE.md                   # Third-party credits
└── README.md
~~~

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| Very slow generation | Enable a GPU: *Runtime → Change runtime type → T4 GPU*. |
| Public link does not appear | Wait for the model to finish loading (the first run downloads it), or re-run Cell 2. |
| "Processing failed" message | Check that the script is not empty and the reference clip is valid, then retry. The activity log and `studio.log` show the details. |
| Out of memory | Lower the quality value or shorten the script. |
| Accent in the output | Use a reference clip in the same language as the script. |

## ⚠️ Responsible use

Do not clone anyone's voice without their explicit permission. Do not use this tool for fraud, impersonation, harassment, or any illegal purpose. You are fully responsible for how you use it.

## 🙏 Credits

Built with open-source components. See [NOTICE.md](NOTICE.md) for details and licenses.

---

<div align="center">Made by <b>AS STUDIO PRO</b></div>
