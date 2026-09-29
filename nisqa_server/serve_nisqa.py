"""
Eigenständiger NISQA-Server für die Perfect Dataset GUI.

Läuft bewusst getrennt vom Hauptprozess (eigene requirements.txt),
da NISQA eine eigene Torch-Umgebung mitbringt und so unabhängig von
Demucs/whisperX/DeepFilterNet aktualisiert/neu gestartet werden kann.

Start: uvicorn serve_nisqa:app --port 8050
(siehe start.sh im Projekt-Wurzelverzeichnis für den kombinierten Start)
"""
import gc
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI(title="NISQA-Server (Perfect Dataset GUI)")

_model = None
_patched = False


def _ensure_torchaudio_compat():
    """Eigenständige Mini-Variante des Kompat-Shims aus dem Hauptprojekt
    (modules/torchaudio_compat.py) — läuft hier in einer eigenen venv,
    kann also nicht direkt darauf zugreifen. torchaudio.load() nutzen wir
    unten gar nicht mehr (soundfile direkt), aber nisqalib selbst könnte
    beim Import noch andere entfernte torchaudio-Attribute referenzieren."""
    global _patched
    if _patched:
        return
    import torch
    import torchaudio

    # Gleiche Begründung wie im Hauptprojekt (modules/torchaudio_compat.py):
    # torch >=2.6 default weights_only=True bricht ältere Checkpoints (hier:
    # das NISQA-Modell). Quelle ist vertrauenswürdig (offizielles NISQA-Modell).
    if not getattr(torch.load, "_pdg_patched", False):
        _original_load = torch.load

        def _patched_load(*args, **kwargs):
            # Erzwingen statt vorbelegen — siehe modules/torchaudio_compat.py
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
    _patched = True


def _load_model():
    """Lazy-Load, damit der Server sofort hochfährt und das Modell erst
    beim ersten Request lädt (und damit auch erst dann Gewichte holt)."""
    global _model
    if _model is None:
        _ensure_torchaudio_compat()
        import nisqalib
        _model = nisqalib.NisqaModel("nisqa")
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/release")
def release():
    """Modell explizit aus dem GPU-Speicher entfernen. Wird vom Client NICHT
    nach jeder einzelnen Bewertung aufgerufen (NISQA läuft oft für 50-100+
    Clips in einem Rutsch — Neuladen pro Clip wäre viel zu teuer), sondern
    einmal am Ende eines kompletten Bewertungs-Durchlaufs (siehe
    modules/quality_score.py, score_all()). Gleiches Grundprinzip wie die
    automatische Freigabe bei separator_server/denoise_server, nur mit
    explizitem Aufruf statt pro Request, weil hier die Aufruf-Häufigkeit
    eine andere ist."""
    global _model
    _model = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"status": "released"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    import torch

    model = _load_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(await file.read())
        tmp.flush()

        # soundfile statt torchaudio.load() — gleiches Muster wie im
        # Hauptprojekt, robust gegen die torchaudio-API-Umbrüche.
        audio, sr = sf.read(tmp.name, always_2d=True, dtype="float32")
        waveform = torch.from_numpy(np.ascontiguousarray(audio.T))
        result = model.predict(waveform, sr)

    mos = float(result["mos_pred"])
    return JSONResponse({"mos": mos})
