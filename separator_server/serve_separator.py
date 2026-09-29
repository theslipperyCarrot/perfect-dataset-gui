"""
Eigenständiger Vocal-Separation-Server über das Paket "audio-separator"
(nutzt UVR-Modelle: MDX-Net, VR-Arch, BS-Roformer — in der Community
etabliert als klar bessere Vocal/Musik-Trennung als Demucs bei vielen
realen Aufnahmen, insbesondere bei durchmischter/leiser Hintergrundmusik).

Läuft als SEPARATER Server (eigene venv), gleiches Muster wie nisqa_server/
und denoise_server/: audio-separator braucht numpy>=2 und eine eigene
onnxruntime/Modell-Download-Kette, die nicht zwingend mit den übrigen
Projekt-Abhängigkeiten in derselben Umgebung koexistieren muss.

Modelle werden beim ersten Aufruf automatisch heruntergeladen (mehrere
hundert MB, je nach Modell) und danach in model_file_dir zwischengespeichert
-> erster Aufruf pro Modell braucht Internet und dauert entsprechend länger.

Start: uvicorn serve_separator:app --port 8052
(siehe start.sh im Projekt-Wurzelverzeichnis für den kombinierten Start)
"""
import gc
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

app = FastAPI(title="Vocal-Separation-Server (Perfect Dataset GUI)")

DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

_separator = None
_loaded_model = None


def _get_separator(model_filename: str, output_dir: str):
    """Lazy-Load, damit der Server sofort hochfährt und Modelle erst beim
    ersten Request geladen (bzw. heruntergeladen) werden.

    output_dir wird bei JEDEM Aufruf explizit gesetzt — sowohl auf dem
    Separator-Objekt als auch (falls schon geladen) auf dem internen
    model_instance. audio-separator übernimmt output_dir nämlich nur beim
    load_model()-Aufruf in den model_instance; wird das Modell (wie hier
    absichtlich) zwischen Requests im Speicher behalten, bliebe
    model_instance.output_dir sonst dauerhaft auf dem Verzeichnis vom
    allerersten Request stehen — die Datei würde dann erfolgreich getrennt,
    aber am falschen Ort landen (genau der Bug, der den FileNotFoundError
    beim Zippen ausgelöst hat)."""
    global _separator, _loaded_model
    from audio_separator.separator import Separator

    if _separator is None:
        _separator = Separator(output_format="WAV")
    _separator.output_dir = output_dir
    if _loaded_model != model_filename:
        _separator.load_model(model_filename=model_filename)
        _loaded_model = model_filename
    if _separator.model_instance is not None:
        _separator.model_instance.output_dir = output_dir
    return _separator


def _release_model():
    """Modell nach JEDER Anfrage wieder aus dem GPU-Speicher entfernen,
    statt es dauerhaft resident zu halten (wie es bis 0.20.3 der Fall war).

    Grund: BS-Roformer (639MB Checkpoint) + der eigene CUDA-Kontext dieses
    Server-Prozesses belegen sonst dauerhaft VRAM, auch lange nachdem die
    Trennung fertig ist — das fehlt dann whisperX/pyannote in Tab 3, wenn
    beide Prozesse gleichzeitig auf derselben GPU laufen: "CUDA failed with
    error out of memory", obwohl UVR selbst längst fertig war.

    Trade-off: der nächste Aufruf braucht wieder ca. 10-15s zum Neuladen
    (siehe Log: "Load model duration: 00:00:11") — gegenüber der eigentlichen
    Trennung (mehrere Minuten) vernachlässigbar, aber der entscheidende
    Unterschied für Tab 3 direkt danach."""
    global _separator, _loaded_model
    if _separator is not None:
        _separator.model_instance = None
    _separator = None
    _loaded_model = None
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


@app.post("/separate")
async def separate(file: UploadFile = File(...), model: str = Form(DEFAULT_MODEL)):
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            in_path = tmp_dir_path / "input.wav"
            with open(in_path, "wb") as fh:
                fh.write(await file.read())

            separator = _get_separator(model, str(tmp_dir_path))
            output_files = separator.separate(str(in_path))

            # audio-separator benennt Ausgabedateien mit "(Vocals)"/"(Instrumental)"
            # im Dateinamen (siehe separator.py, STEM_NAME_MAP) — wir erkennen die
            # Vocals-Datei am Namen und packen beide Stems in ein ZIP, damit der
            # Client (modules/uvr_separation.py) beide auf einmal bekommt (für den
            # "Musik-Abschnitte rausschneiden"-Modus wird auch die Instrumental-
            # Spur gebraucht, nicht nur die Vocals).
            zip_path = tmp_dir_path / "result.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                for f in output_files:
                    fp = Path(f)
                    if not fp.is_absolute():
                        fp = tmp_dir_path / fp
                    if not fp.exists():
                        raise HTTPException(
                            status_code=500,
                            detail=(
                                f"audio-separator hat '{fp}' als Ausgabedatei gemeldet, "
                                f"die Datei existiert dort aber nicht (Modell: {model}). "
                                "Das deutet auf ein output_dir-Sync-Problem hin, siehe "
                                "_get_separator()."
                            ),
                        )
                    label = "vocals" if "vocal" in fp.stem.lower() else "instrumental"
                    zf.write(fp, arcname=f"{label}.wav")

            out_path = tempfile.NamedTemporaryFile(suffix=".zip", delete=False).name
            shutil.copy(zip_path, out_path)

        return FileResponse(out_path, media_type="application/zip")
    finally:
        # IMMER ausführen, auch bei Fehlern mittendrin — sonst bliebe das
        # Modell nach einem Fehlerfall erst recht dauerhaft im GPU-Speicher.
        _release_model()
