import functools, os, re, time, glob, zipfile, logging, traceback
import numpy as np, soundfile as sf, gradio as gr
from engine import Engine, SR

APP = "AS VOICE STUDIO V3"
BASE = os.getenv("AS_HOME", "/content" if os.path.isdir("/content") else os.path.expanduser("~/as_voice_studio"))
LIB, OUT = f"{BASE}/voices", f"{BASE}/exports"
os.makedirs(LIB, exist_ok=True); os.makedirs(OUT, exist_ok=True)
logging.basicConfig(filename=f"{BASE}/studio.log", level=logging.ERROR)
print(f"Starting {APP} ...")
eng = Engine()
print("Ready.")

TAGS = ["[laughter]", "[sigh]", "[confirmation-en]", "[question-en]", "[question-ah]", "[question-oh]",
        "[question-ei]", "[question-yi]", "[surprise-ah]", "[surprise-oh]", "[surprise-wa]",
        "[surprise-yo]", "[dissatisfaction-hnn]"]
GENDER = ["Any", "male", "female"]
AGE = ["Any", "child", "teenager", "young adult", "middle-aged", "elderly"]
PITCH = ["Any", "very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"]
STYLE = ["Any", "whisper"]
ACCENT = ["Any", "american accent", "british accent", "australian accent", "canadian accent", "indian accent"]
PRESETS = {"Custom": ("Any",) * 5,
           "Narrator": ("male", "middle-aged", "low pitch", "Any", "Any"),
           "News Anchor": ("female", "young adult", "moderate pitch", "Any", "american accent"),
           "Storyteller": ("female", "elderly", "low pitch", "Any", "british accent"),
           "Presenter": ("male", "young adult", "high pitch", "Any", "Any"),
           "Soft Whisper": ("female", "young adult", "Any", "whisper", "Any")}


# ---------- core ----------
LABELS = {"do_clone": "Voice Clone", "do_design": "Voice Design", "do_auto": "Auto Voice",
          "do_expr": "Expressions", "do_lib": "Voice Library", "do_batch": "Batch Export"}

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def safe(fn):
    @functools.wraps(fn)
    def w(*a, **k):
        name, t0 = LABELS.get(fn.__name__, fn.__name__), time.time()
        log(f"> {name}: started")
        try:
            r = fn(*a, **k)
            log(f"OK {name}: done in {time.time()-t0:.1f}s")
            return r
        except gr.Error as e:
            log(f"!! {name}: {e}")
            raise gr.Error(str(e.message), print_exception=False)
        except Exception as e:
            logging.error(traceback.format_exc())
            log(f"!! {name}: failed ({type(e).__name__}: {e})")
            raise gr.Error("Processing failed. Please check your inputs and try again.", print_exception=False)
    return w

def split_text(t, n=350):
    out, cur = [], ""
    for p in re.split(r"(?<=[.!?\u06d4\u061f\u0964])\s+|\n+", t.strip()):
        if cur and len(cur) + len(p) > n: out.append(cur); cur = p
        else: cur = f"{cur} {p}".strip()
    return out + ([cur] if cur else [])

def synth(text, steps, speed, **kw):
    if not (text or "").strip(): raise gr.Error("Please enter some text.")
    t0, gap, parts = time.time(), np.zeros(int(0.15 * SR), np.float32), []
    for c in split_text(text):
        parts += [eng.generate(c, steps, speed, **kw), gap]
    wav = np.concatenate(parts[:-1])
    return (SR, wav), f"Generated {len(wav)/SR:.1f}s of audio in {time.time()-t0:.1f}s."

def names(): return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{LIB}/*.pt"))
def load(n):
    if not n: raise gr.Error("Select a saved voice first.")
    return eng.load_prompt(f"{LIB}/{n}.pt")
def prompt_from(ref, rt):
    if not ref: raise gr.Error("Please add a reference recording (3-10 seconds).")
    return eng.make_prompt(ref, (rt or "").strip() or None)
def instr(*p): return ", ".join(x for x in p if x and x != "Any")
def refresh(): return gr.update(choices=names())


# ---------- handlers ----------
@safe
def do_clone(text, ref, rt, steps, speed, save_as):
    p, note = prompt_from(ref, rt), ""
    n = "".join(c for c in (save_as or "") if c.isalnum() or c in "-_ ").strip()
    if n: eng.save_prompt(p, f"{LIB}/{n}.pt"); note = f" Voice saved as '{n}'."
    a, m = synth(text, steps, speed, prompt=p)
    return a, m + note, refresh(), refresh()

@safe
def do_design(text, g, a, p, s, ac, extra, steps, speed):
    ins = instr(g, a, p, s, ac, extra)
    if not ins: raise gr.Error("Choose at least one voice attribute.")
    return synth(text, steps, speed, instruct=ins)

@safe
def do_auto(text, steps, speed): return synth(text, steps, speed)

@safe
def do_expr(text, mode, ins, steps, speed):
    return synth(text, steps, speed, instruct=(ins or "").strip() or None if mode == "Designed voice" else None)

@safe
def do_lib(n, text, steps, speed): return synth(text, steps, speed, prompt=load(n))

def do_del(n):
    if n and os.path.exists(f"{LIB}/{n}.pt"): os.remove(f"{LIB}/{n}.pt")
    return refresh(), refresh(), f"Deleted '{n}'." if n else ""

@safe
def do_batch(lines, mode, lib, ins, steps, speed, progress=gr.Progress()):
    items = [l.strip() for l in (lines or "").splitlines() if l.strip()]
    if not items: raise gr.Error("Enter at least one line of text.")
    kw = {}
    if mode == "Saved voice": kw["prompt"] = load(lib)
    if mode == "Designed voice":
        if not (ins or "").strip(): raise gr.Error("Enter voice attributes.")
        kw["instruct"] = ins.strip()
    zp, total = f"{OUT}/batch_{time.strftime('%Y%m%d_%H%M%S')}.zip", 0.0
    with zipfile.ZipFile(zp, "w") as z:
        for i, line in enumerate(progress.tqdm(items, desc="Rendering"), 1):
            w = np.concatenate([eng.generate(c, steps, speed, **kw) for c in split_text(line)])
            f = f"{OUT}/track_{i:03d}.wav"; sf.write(f, w, SR); z.write(f, os.path.basename(f)); os.remove(f)
            total += len(w) / SR
    return zp, f"Exported {len(items)} tracks, {total:.1f}s total."

def sysinfo():
    return f"**Compute:** {eng.device}  \n**Saved voices:** {len(names())}  \n**Sample rate:** {SR//1000} kHz"


# ---------- design ----------
CSS = """
:root,.dark,.gradio-container{--body-background-fill:#f5f6fa;--background-fill-primary:#ffffff;--background-fill-secondary:#f1f3f9;
--block-background-fill:#ffffff;--panel-background-fill:#ffffff;--input-background-fill:#f8f9fc;--input-background-fill-focus:#ffffff;
--border-color-primary:#e3e7ef;--block-border-color:#e3e7ef;--input-border-color:#d9dfeb;--body-text-color:#1c2333;
--body-text-color-subdued:#667085;--block-label-text-color:#667085;--block-title-text-color:#344054;--color-accent:#6366f1;
--color-accent-soft:#eef0ff;--button-secondary-background-fill:#eef1f7;--button-secondary-background-fill-hover:#e2e7f1;
--button-secondary-text-color:#1c2333;--button-secondary-border-color:#d9dfeb;--loader-color:#6366f1;--slider-color:#6366f1;
--neutral-50:#f8f9fc;--neutral-100:#f1f3f9;--neutral-200:#e3e7ef;--shadow-drop:none;--block-shadow:none;color-scheme:light}
body,.gradio-container{background:#f5f6fa!important;color:#1c2333!important;font-family:Inter,system-ui,sans-serif!important}
.gradio-container{max-width:1120px!important}
footer,.built-with,.show-api{display:none!important}
#hero{display:flex;flex-direction:column;gap:6px;padding:26px 28px;border:1px solid #e3e7ef;border-left:4px solid #6366f1;border-radius:14px;background:#ffffff;margin:4px 0 14px;box-shadow:0 2px 10px rgba(28,35,51,.05)}
#hero h1{margin:0;font-size:1.7rem;font-weight:700;letter-spacing:.04em;color:#111827}
#hero h1 span{color:#4f46e5}
#hero p{margin:0;color:#667085;font-size:.95rem}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.chips i{font-style:normal;font-size:.75rem;color:#4338ca;border:1px solid #d9dcfb;background:#eef0ff;padding:3px 11px;border-radius:999px}
.tab-wrapper,.tab-container{border-color:#e3e7ef!important}
button[role=tab]{font-weight:600!important;color:#667085!important}
button[role=tab][aria-selected=true]{color:#4338ca!important;border-color:#6366f1!important}
.overflow-dropdown,.overflow-menu .overflow-dropdown{background:#ffffff!important;border:1px solid #e3e7ef!important;border-radius:10px!important;box-shadow:0 8px 24px rgba(28,35,51,.12)!important}
.overflow-dropdown button{color:#344054!important;background:transparent!important}
.overflow-dropdown button:hover{background:#f1f3f9!important}
.block,.form,.gr-group{border-radius:12px!important}
audio,.waveform-container,.audio-container,.component-wrapper{background:#ffffff!important}
.generating,.pending,.wrap.default,.wrap{background:transparent!important;color:#475467!important}
button.primary{background:linear-gradient(135deg,#6366f1,#4f46e5)!important;color:#fff!important;border:0!important;font-weight:600!important;border-radius:10px!important;box-shadow:0 4px 14px rgba(99,102,241,.3)!important}
button.primary:hover{filter:brightness(1.08)}
textarea,input{color:#1c2333!important}
.empty .icon,.empty svg,[aria-label='Empty value'] svg{display:none!important}
#foot{text-align:center;color:#98a2b3;font-size:.8rem;padding:14px 0}
@media(max-width:640px){#hero{padding:18px}#hero h1{font-size:1.25rem}}
"""
FORCE_DARK = "() => { document.body.classList.remove('dark'); }"

def qs(v=32):
    with gr.Row():
        s = gr.Slider(8, 64, value=v, step=1, label="Quality", info="16 fast · 32 balanced · 48+ studio")
        sp = gr.Slider(0.5, 1.8, value=1.0, step=0.05, label="Speaking speed")
    return s, sp

def out_col(file=False):
    a = gr.Audio(label="Result", type="numpy", interactive=False) if not file else gr.File(label="Download")
    return a, gr.Markdown()

with gr.Blocks(css=CSS, js=FORCE_DARK, title=APP, theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate")) as demo:
    gr.HTML(f'<div id="hero"><h1>AS VOICE <span>STUDIO</span> V3</h1>'
            '<p>Multilingual voice cloning, design and production in one workspace.</p>'
            '<div class="chips"><i>600+ languages</i><i>Voice cloning</i><i>Voice design</i><i>Batch export</i></div></div>')
    with gr.Tabs():
        with gr.Tab("Voice Clone"):
            with gr.Row():
                with gr.Column():
                    c_t = gr.Textbox(lines=5, label="Script", placeholder="Type or paste the text to speak...")
                    c_r = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Reference voice (3-10 seconds)")
                    c_rt = gr.Textbox(label="Reference transcript (optional)", info="Leave empty to transcribe automatically.")
                    c_n = gr.Textbox(label="Save voice as (optional)")
                    c_s, c_sp = qs()
                    c_b = gr.Button("Generate", variant="primary")
                with gr.Column(): c_o, c_m = out_col()
        with gr.Tab("Voice Design"):
            with gr.Row():
                with gr.Column():
                    d_t = gr.Textbox(lines=5, label="Script")
                    d_pr = gr.Dropdown(list(PRESETS), value="Custom", label="Preset")
                    with gr.Row():
                        d_g = gr.Dropdown(GENDER, value="female", label="Gender"); d_a = gr.Dropdown(AGE, value="Any", label="Age")
                    with gr.Row():
                        d_p = gr.Dropdown(PITCH, value="Any", label="Pitch"); d_st = gr.Dropdown(STYLE, value="Any", label="Style")
                    d_ac = gr.Dropdown(ACCENT, value="Any", label="Accent")
                    d_x = gr.Textbox(label="Additional attributes (optional)", info="Comma separated.")
                    d_s, d_sp = qs()
                    d_b = gr.Button("Generate", variant="primary")
                with gr.Column(): d_o, d_m = out_col()
        with gr.Tab("Auto Voice"):
            with gr.Row():
                with gr.Column():
                    a_t = gr.Textbox(lines=7, label="Script"); a_s, a_sp = qs(); a_b = gr.Button("Generate", variant="primary")
                with gr.Column(): a_o, a_m = out_col()
        with gr.Tab("Expressions"):
            with gr.Row():
                with gr.Column():
                    e_t = gr.Textbox(lines=5, label="Script", value="[laughter] You really got me. I did not see that coming.")
                    with gr.Row():
                        e_tg = gr.Dropdown(TAGS, value=TAGS[0], label="Expression"); e_add = gr.Button("Insert into script")
                    e_md = gr.Radio(["Auto voice", "Designed voice"], value="Auto voice", label="Voice")
                    e_i = gr.Textbox(label="Voice attributes", placeholder="female, low pitch, british accent")
                    e_s, e_sp = qs(); e_b = gr.Button("Generate", variant="primary")
                with gr.Column():
                    e_o, e_m = out_col()
                    gr.Markdown("English pronunciation can be corrected with phonemes, e.g. `[B EY1 S]`.")
        with gr.Tab("Voice Library"):
            with gr.Row():
                with gr.Column():
                    l_n = gr.Dropdown(names(), label="Saved voices")
                    with gr.Row(): l_r = gr.Button("Refresh"); l_d = gr.Button("Delete")
                    l_t = gr.Textbox(lines=5, label="Script"); l_s, l_sp = qs(); l_b = gr.Button("Generate", variant="primary")
                with gr.Column(): l_o, l_m = out_col()
        with gr.Tab("Batch Export"):
            with gr.Row():
                with gr.Column():
                    b_t = gr.Textbox(lines=8, label="Scripts", info="One line per audio file.")
                    b_md = gr.Radio(["Auto voice", "Designed voice", "Saved voice"], value="Auto voice", label="Voice")
                    b_l = gr.Dropdown(names(), label="Saved voice"); b_i = gr.Textbox(label="Voice attributes")
                    b_s, b_sp = qs(16); b_b = gr.Button("Export ZIP", variant="primary")
                with gr.Column(): b_o, b_m = out_col(True)
        with gr.Tab("Guide"):
            gi = gr.Markdown(sysinfo()); gr.Button("Refresh status").click(sysinfo, None, gi)
            gr.Markdown("""**Tips**
- Use a clean 3-10 second reference clip without background noise.
- Keep the reference and target language the same to avoid an accent.
- Long scripts are split automatically at sentence boundaries.
- Write numbers as words for best pronunciation.

**Responsible use:** clone only your own voice or voices you have permission to use. Impersonation and fraud are prohibited.""")
    gr.HTML(f'<div id="foot">{APP}</div>')

    c_b.click(do_clone, [c_t, c_r, c_rt, c_s, c_sp, c_n], [c_o, c_m, l_n, b_l])
    d_pr.change(lambda k: list(PRESETS[k]), d_pr, [d_g, d_a, d_p, d_st, d_ac])
    d_b.click(do_design, [d_t, d_g, d_a, d_p, d_st, d_ac, d_x, d_s, d_sp], [d_o, d_m])
    a_b.click(do_auto, [a_t, a_s, a_sp], [a_o, a_m])
    e_add.click(lambda t, g: f"{(t or '').rstrip()} {g}".strip(), [e_t, e_tg], e_t)
    e_b.click(do_expr, [e_t, e_md, e_i, e_s, e_sp], [e_o, e_m])
    l_b.click(do_lib, [l_n, l_t, l_s, l_sp], [l_o, l_m])
    l_r.click(refresh, None, l_n).then(refresh, None, b_l)
    l_d.click(do_del, l_n, [l_n, b_l, l_m])
    b_b.click(do_batch, [b_t, b_md, b_l, b_i, b_s, b_sp], [b_o, b_m])

if __name__ == "__main__":
    colab = os.path.isdir("/content")
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        _, local_url, share_url = demo.queue().launch(
            share=colab, inline=False, quiet=True, prevent_thread_lock=True,
            show_api=os.getenv("AS_DEBUG") == "1", server_name=None if colab else "0.0.0.0")
    if colab and not share_url:
        print("Public link could not be created. Stop this cell and run it again.", flush=True)
    print(f"* Running on {'public' if share_url else 'local'} URL: {share_url or local_url}\n", flush=True)
    demo.block_thread()
