"""
Optionales Denoising mit DeepFilterNet.

Demucs trennt Musik/Instrumente von Sprache, entfernt aber kein
Mikro-Hiss, Raumhall oder Lüfter-/Grundrauschen in der Sprachspur
selbst. DeepFilterNet ist ein leichtgewichtiges, echtzeitfähiges
Modell genau dafür.

Läuft als SEPARATER Server (denoise_server/, eigene venv) — DeepFilterNet
(max. Version 0.5.6) braucht zwingend numpy<2.0, was mit modernen
whisperx/ctranslate2-Versionen (numpy>=2.0) im Hauptprojekt nicht in
derselben Umgebung koexistieren kann. Client hier ruft den Server nur
per HTTP auf, mit Fallback (Original unverändert zurückgeben), falls der
Server mal nicht erreichbar ist — bricht die Pipeline dadurch nicht ab.
"""
import sys
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DENOISE_SERVER_URL


def denoise_file(input_path: Path, output_path: Path, timeout: float = 60.0) -> Path:
    """Entfernt Restrauschen aus input_path und schreibt nach output_path.
    Bei nicht erreichbarem Denoise-Server wird die Eingabedatei unverändert
    nach output_path kopiert (Pipeline läuft weiter, nur ohne Denoising)."""
    try:
        with open(input_path, "rb") as fh:
            resp = requests.post(DENOISE_SERVER_URL, files={"file": fh}, timeout=timeout)
        resp.raise_for_status()
        with open(output_path, "wb") as out:
            out.write(resp.content)
        return output_path
    except Exception as e:
        print(f"[denoise] Denoise-Server nicht erreichbar/Fehler ({e}) — "
              f"nutze unbearbeitetes Audio für {input_path.name}.")
        import shutil
        shutil.copy(input_path, output_path)
        return output_path
