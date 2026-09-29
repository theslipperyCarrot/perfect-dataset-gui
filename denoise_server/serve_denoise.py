"""
Eigenständiger DeepFilterNet-Denoise-Server für die Perfect Dataset GUI.

Läuft bewusst getrennt vom Hauptprozess: DeepFilterNet (aktuell max. 0.5.6)
braucht zwingend numpy<2.0, was mit modernen whisperx/ctranslate2-Versionen
(numpy>=2.0) im Hauptprojekt nicht in derselben Umgebung koexistieren kann.

Start: uvicorn serve_denoise:app --port 8051
(siehe start.sh im Projekt-Wurzelverzeichnis für den kombinierten Start)
"""
import gc
import sys
import types
import importlib.abc
import importlib.machinery
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

app = FastAPI(title="Denoise-Server (Perfect Dataset GUI)")

_model = None
_df_state = None
_patched = False


class _DummyAny:
    def __call__(self, *a, **k):
        return _DummyAny()

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _DummyAny()


class _PermissiveModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _DummyAny()


class _TorchaudioBackendShimLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = _PermissiveModule(spec.name)
        mod.__path__ = []
        return mod

    def exec_module(self, module):
        pass


class _TorchaudioBackendShimFinder(importlib.abc.MetaPathFinder):
    PREFIX = "torchaudio.backend"

    def find_spec(self, fullname, path, target=None):
        if fullname == self.PREFIX or fullname.startswith(self.PREFIX + "."):
            return importlib.machinery.ModuleSpec(
                fullname, _TorchaudioBackendShimLoader(), is_package=True,
            )
        return None


def _ensure_torchaudio_compat():
    """Eigenständige Variante des Kompat-Shims aus dem Hauptprojekt
    (modules/torchaudio_compat.py) — läuft hier in einer eigenen venv.
    WICHTIG: braucht auch den Meta-Path-Finder für 'torchaudio.backend.*'
    (nicht nur einzelne Attribute), da df/io.py explizit
    'from torchaudio.backend.common import AudioMetaData' importiert."""
    global _patched
    if _patched:
        return
    import torch
    import torchaudio

    if not getattr(torch.load, "_pdg_patched", False):
        _original_load = torch.load

        def _patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _original_load(*args, **kwargs)

        _patched_load._pdg_patched = True
        torch.load = _patched_load

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda name=None: None
    if not hasattr(torchaudio, "AudioMetaData"):
        class _AudioMetaDataShim:
            pass
        torchaudio.AudioMetaData = _AudioMetaDataShim

    try:
        import torchaudio.backend  # noqa: F401
    except ModuleNotFoundError:
        if not any(isinstance(f, _TorchaudioBackendShimFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, _TorchaudioBackendShimFinder())

    _patched = True


def _load_model():
    """Lazy-Load, damit der Server sofort hochfährt und das Modell erst
    beim ersten Request lädt (und damit auch erst dann Gewichte holt)."""
    global _model, _df_state
    if _model is None:
        _ensure_torchaudio_compat()
        from df.enhance import init_df
        _model, _df_state, _ = init_df()
    return _model, _df_state


def _release_model():
    """Modell nach JEDER Anfrage wieder aus dem GPU-Speicher entfernen,
    statt es dauerhaft resident zu halten (gleiches Prinzip wie bei
    separator_server ab 0.20.4). DeepFilterNet3 ist klein und lädt sehr
    schnell (deutlich unter 1s) — die Kosten für erneutes Laden pro Datei
    sind vernachlässigbar, der freigewordene VRAM zählt hier mehr."""
    global _model, _df_state
    _model = None
    _df_state = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/denoise")
async def denoise(file: UploadFile = File(...)):
    try:
        import torch
        _ensure_torchaudio_compat()  # MUSS vor 'from df.enhance import enhance' passieren
        from df.enhance import enhance

        model, df_state = _load_model()
        target_sr = df_state.sr()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_in:
            tmp_in.write(await file.read())
            tmp_in.flush()

            audio, sr = sf.read(tmp_in.name, always_2d=True, dtype="float32")
            audio = audio.T  # (Samples, Kanäle) -> (Kanäle, Samples)
            if sr != target_sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, axis=-1)
            if audio.shape[0] > 1:
                audio = audio.mean(axis=0, keepdims=True)

            wav_tensor = torch.from_numpy(np.ascontiguousarray(audio))
            enhanced = enhance(model, df_state, wav_tensor)
            enhanced_np = enhanced.detach().cpu().numpy()
            if enhanced_np.ndim == 2:
                enhanced_np = enhanced_np[0]

        out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(out_path, enhanced_np, target_sr, subtype="PCM_16")
        return FileResponse(out_path, media_type="audio/wav")
    finally:
        _release_model()
