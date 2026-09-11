"""
Client für den separaten NISQA-Server (siehe nisqa_server/).
Läuft bewusst als eigener Prozess (eigene Torch-Umgebung, per
start.sh zusammen mit der PD-GUI gestartet) statt eingebettet in
diesen Prozess.
"""
import sys
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SEGMENTS_DIR, NISQA_SERVER_URL
from modules.segment_and_transcribe import ClipEntry, load_manifest, save_manifest


def score_clip(clip_path: Path, timeout: float = 120.0) -> float:
    """Gibt den NISQA-MOS (1-5) zurück, oder None falls der Server nicht
    erreichbar ist / einen Fehler meldet — bricht den Review-Flow dadurch
    nicht ab, sondern lässt den Score einfach leer."""
    try:
        with open(clip_path, "rb") as fh:
            resp = requests.post(NISQA_SERVER_URL, files={"file": fh}, timeout=timeout)
        resp.raise_for_status()
        return float(resp.json()["mos"])
    except Exception as e:
        print(f"[quality_score] NISQA-Server nicht erreichbar/Fehler für {clip_path.name}: {e}")
        return None


def score_all(progress_cb=None) -> tuple[int, int]:
    """Bewertet alle Clips im Manifest. Gibt (bewertet, fehlgeschlagen) zurück."""
    entries: list[ClipEntry] = load_manifest()
    scored, failed = 0, 0
    total = len(entries)
    for i, e in enumerate(entries, start=1):
        mos = score_clip(SEGMENTS_DIR / e.clip_filename)
        if mos is not None:
            e.nisqa_mos = mos
            scored += 1
        else:
            failed += 1
        if progress_cb:
            progress_cb(i, total, e.id)
    save_manifest(entries)
    return scored, failed
