"""
AS VOICE STUDIO V3
-------------------
A restyled, professional front-end for the OmniVoice engine (k2-fsa/OmniVoice).
All generation logic (model, voice clone, voice design, subtitles, silence
removal, non-verbal tags, 600+ languages) is reused as-is from the OmniVoice
reference app.py — only the interface, branding and layout are new.

Must be run with cwd = the omnivoice-colab clone directory (the folder that
contains app.py and the OmniVoice/ subfolder), e.g.:
    cd /content/omnivoice-colab && python as_voice_studio_app.py
"""

import os
import sys
import importlib.util
import gradio as gr

_here = os.getcwd()
_spec = importlib.util.spec_from_file_location("omnivoice_engine", os.path.join(_here, "app.py"))
ov = importlib.util.module_from_spec(_spec)
sys.modules["omnivoice_engine"] = ov
_spec.loader.exec_module(ov)

print("AS VOICE STUDIO V3 — engine loaded, building custom interface...")

BRAND_THEME = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="violet",
    neutral_hue="slate",
)

BRAND_CSS = ov.css + """
#as-header {
    text-align: center; margin: 10px auto 24px auto; max-width: 900px;
    padding: 26px 20px; border-radius: 16px;
    background: linear-gradient(135deg, #1f1147 0%, #3a1c71 45%, #6b2b8c 100%);
    color: #f5f2ff; box-shadow: 0 10px 26px rgba(0,0,0,.3);
}
#as-header h1 { font-size: 2.3em; margin: 0 0 6px 0; letter-spacing: .5px; color: #ffffff !important; }
#as-header p { margin: 0; opacity: .9; font-size: 1.05em; color: #f0e9ff !important; }
#as-footer { text-align: center; opacity: .6; font-size: .85em; margin-top: 18px; }
.as-card { border-radius: 14px !important; }
"""

EVENT_TAGS = ov.EVENT_TAGS
INSERT_TAG_JS_VC = ov.INSERT_TAG_JS_VC
INSERT_TAG_JS_VD = ov.INSERT_TAG_JS_VD
CATEGORIES = ov._CATEGORIES
ATTR_INFO = ov._ATTR_INFO
lang_dropdown = ov._lang_dropdown
gen_settings = ov._gen_settings
gen_core = ov._gen_core
tts_file_name = ov.tts_file_name
generate_subtitles_if_needed = ov.generate_subtitles_if_needed
subtitle_maker = ov.subtitle_maker

import scipy.io.wavfile as wavfile


def clone_fn(text, lang, ref_aud, ref_text, want_subs, ns, gs, dn, sp, du, pp, po):
    res = gen_core(text, lang, ref_aud, None, ns, gs, dn, sp, du, pp, po, mode="clone", ref_text=ref_text)
    if res[0] is None:
        return None, res[1], None, None, None, None
    audio_tuple, status = res
    sr, waveform = audio_tuple
    tmp_wav = tts_file_name(text, language=lang)
    wavfile.write(tmp_wav, sr, waveform)
    c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)
    return audio_tuple, status, tmp_wav, c_srt, w_srt, s_srt


def auto_transcribe(audio_path, lang):
    if not audio_path:
        return gr.update(value="")
    try:
        whisper_lang = lang if lang != "Auto" else None
        whisper_results = subtitle_maker(audio_path, whisper_lang)
        if whisper_results and len(whisper_results) > 7:
            return gr.update(value=whisper_results[7])
    except Exception as e:
        print(f"Auto-transcription failed: {e}")
    return gr.update(value="")


def build_instruct(groups):
    selected = [g for g in groups if g and g != "Auto"]
    if not selected:
        return None
    return ", ".join([ov.DIALECT_MAP.get(v, v) for v in selected])


def design_fn(text, lang, want_subs, ns, gs, dn, sp, du, pp, po, *groups):
    instruct = build_instruct(groups)
    res = gen_core(text, lang, None, instruct, ns, gs, dn, sp, du, pp, po, mode="design")
    if res[0] is None:
        return None, res[1], None, None, None, None
    audio_tuple, status = res
    sr, waveform = audio_tuple
    tmp_wav = tts_file_name(text, language=lang)
    wavfile.write(tmp_wav, sr, waveform)
    c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)
    return audio_tuple, status, tmp_wav, c_srt, w_srt, s_srt


def event_tag_row(target_textbox, js):
    with gr.Row(elem_classes=["tag-container"]):
        for tag in EVENT_TAGS:
            btn = gr.Button(tag, elem_classes=["tag-btn"])
            btn.click(fn=None, inputs=[btn, target_textbox], outputs=target_textbox, js=js)


with gr.Blocks(theme=BRAND_THEME, css=BRAND_CSS, title="AS VOICE STUDIO V3") as demo:
    gr.HTML(
        """
        <div id="as-header">
            <h1>🎙️ AS VOICE STUDIO V3</h1>
            <p>Professional voice cloning & voice design · 600+ languages</p>
        </div>
        """
    )

    with gr.Tabs():
        with gr.TabItem("🎭 Voice Clone"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["as-card"]):
                    vc_text = gr.Textbox(
                        label="Text to Synthesize", lines=4,
                        placeholder="Enter the text to synthesize...",
                        elem_id="vc_textbox",
                    )
                    event_tag_row(vc_text, INSERT_TAG_JS_VC)

                    with gr.Row():
                        vc_lang = lang_dropdown("Language (optional)")
                        vc_want_subs = gr.Checkbox(label="Generate Subtitles", value=False)

                    vc_ref_audio = gr.Audio(
                        label="Reference Voice (3–10 seconds)", type="filepath",
                        elem_classes="compact-audio",
                    )
                    vc_ref_text = gr.Textbox(
                        label="Reference Text", lines=2,
                        placeholder="Auto-transcribed when you upload audio. Edit if Whisper gets it wrong.",
                    )
                    vc_btn = gr.Button("✨ Generate Cloned Voice", variant="primary")
                    vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po = gen_settings()

                with gr.Column(scale=1, elem_classes=["as-card"]):
                    vc_audio = gr.Audio(label="Output Audio", type="numpy")
                    vc_status = gr.Textbox(label="Status", lines=1)
                    with gr.Accordion("Download files", open=False):
                        vc_out_wav = gr.File(label="Generated Audio (WAV)")
                        vc_out_custom_srt = gr.File(label="Sentence Level SRT")
                        vc_out_word_srt = gr.File(label="Word Level SRT")
                        vc_out_shorts_srt = gr.File(label="Shorts SRT")

            vc_ref_audio.change(fn=auto_transcribe, inputs=[vc_ref_audio, vc_lang], outputs=[vc_ref_text])

            vc_btn.click(
                clone_fn,
                inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_want_subs,
                        vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po],
                outputs=[vc_audio, vc_status, vc_out_wav, vc_out_custom_srt, vc_out_word_srt, vc_out_shorts_srt],
            )

        with gr.TabItem("🎨 Voice Design"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["as-card"]):
                    vd_text = gr.Textbox(
                        label="Text to Synthesize", lines=4,
                        placeholder="Enter the text to synthesize...",
                        elem_id="vd_textbox",
                    )
                    event_tag_row(vd_text, INSERT_TAG_JS_VD)

                    with gr.Row():
                        vd_lang = lang_dropdown(value="Auto")
                        vd_want_subs = gr.Checkbox(label="Generate Subtitles", value=False)

                    vd_btn = gr.Button("✨ Generate Designed Voice", variant="primary")

                    with gr.Accordion("Character Voice Design", open=True):
                        vd_groups = []
                        for cat, choices in CATEGORIES.items():
                            default_val = "Auto"
                            if cat == "Gender":
                                default_val = "Female"
                            elif cat == "Age":
                                default_val = "Young Adult"
                            vd_groups.append(
                                gr.Dropdown(
                                    label=cat, choices=["Auto"] + choices, value=default_val,
                                    info=ATTR_INFO.get(cat),
                                )
                            )

                    vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po = gen_settings()

                with gr.Column(scale=1, elem_classes=["as-card"]):
                    vd_audio = gr.Audio(label="Output Audio", type="numpy")
                    vd_status = gr.Textbox(label="Status", lines=1)
                    with gr.Accordion("Download files", open=False):
                        vd_out_wav = gr.File(label="Generated Audio (WAV)")
                        vd_out_custom_srt = gr.File(label="Sentence Level SRT")
                        vd_out_word_srt = gr.File(label="Word Level SRT")
                        vd_out_shorts_srt = gr.File(label="Shorts SRT")

            vd_btn.click(
                design_fn,
                inputs=[vd_text, vd_lang, vd_want_subs, vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po] + vd_groups,
                outputs=[vd_audio, vd_status, vd_out_wav, vd_out_custom_srt, vd_out_word_srt, vd_out_shorts_srt],
            )

    gr.HTML('<div id="as-footer">AS VOICE STUDIO V3 · Use responsibly — do not clone anyone\'s voice without their consent.</div>')


if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
