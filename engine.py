"""Voice engine wrapper. Set AS_DEMO_MODE=1 to run the UI without a GPU/model."""
import os, io, pickle, logging, warnings, contextlib
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import numpy as np

SR = 24000
DEMO = os.getenv("AS_DEMO_MODE") == "1"


class Engine:
    def __init__(self):
        self.device = "demo"
        self.pre = True
        self.post = True
        self.gs = 2.0
        self.denoise = True
        self.lang = None
        self.dur = None
        if DEMO:
            return
        import torch
        from omnivoice import OmniVoice, VoiceClonePrompt
        self._P = VoiceClonePrompt
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=self.device, dtype=dtype, load_asr=True)

    def generate(self, text, steps=32, speed=1.0, prompt=None, instruct=None):
        if DEMO:
            t = np.linspace(0, max(1.0, len(text) * 0.05) / speed, int(SR * max(1.0, len(text) * 0.05) / speed))
            return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        kw = {"text": text, "num_step": int(steps), "speed": float(speed)}
        if prompt is not None: kw["voice_clone_prompt"] = prompt
        if instruct: kw["instruct"] = instruct
        if self.lang: kw["language"] = self.lang
        if self.dur: kw["duration"] = float(self.dur)
        try:
            from omnivoice import OmniVoiceGenerationConfig
            kw["generation_config"] = OmniVoiceGenerationConfig(num_step=int(steps), postprocess_output=bool(self.post),
                guidance_scale=float(self.gs), denoise=bool(self.denoise))
        except Exception:
            pass
        return np.asarray(self.m.generate(**kw)[0], dtype=np.float32).squeeze()

    def make_prompt(self, ref, ref_text=None):
        if DEMO: return {"ref": ref}
        kw = {"ref_audio": ref, "preprocess_prompt": bool(self.pre)}
        if ref_text: kw["ref_text"] = ref_text
        return self.m.create_voice_clone_prompt(**kw)

    def save_prompt(self, p, path):
        if DEMO: pickle.dump(p, open(path, "wb"))
        else: p.save(path)

    def load_prompt(self, path):
        return pickle.load(open(path, "rb")) if DEMO else self._P.load(path)
