"""
Snapshots: sichert den aktuellen data/-Stand (Import/Processed/Segments/
Export) in einen eigenen Ordner unter snapshots/, damit man bei Problemen
zu einem genau definierten Punkt zurückspringen kann. Jeder Snapshot wird
mit der aktuellen PROJECT_VERSION getaggt, damit klar ist, mit welchem
Code-Stand er erzeugt wurde.

Bewusst getrennt vom Reset-Mechanismus (reset.py) und vom Reset-beim-
Beenden (app.py): Snapshots liegen außerhalb von data/ und werden davon
nicht berührt — man kann also vor dem Schließen der GUI (die data/ dann
automatisch leert) gezielt einen Snapshot der guten Ergebnisse ziehen.
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BASE_DIR, RAW_DIR, PROCESSED_DIR, SEGMENTS_DIR, EXPORT_DIR, PROJECT_VERSION

SNAPSHOTS_DIR = BASE_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(exist_ok=True)

DATA_SUBDIRS = {
    "raw": RAW_DIR,
    "processed": PROCESSED_DIR,
    "segments": SEGMENTS_DIR,
    "export": EXPORT_DIR,
}


def _safe_label(label: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in label.strip())
    return cleaned[:40]


def create_snapshot(label: str = "") -> str:
    """Kopiert den kompletten data/-Stand in einen neuen, mit Zeitstempel
    und PROJECT_VERSION benannten Ordner. Gibt den Snapshot-Namen zurück."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{timestamp}_v{PROJECT_VERSION}"
    label_part = _safe_label(label)
    if label_part:
        name += f"_{label_part}"

    target = SNAPSHOTS_DIR / name
    target.mkdir(parents=True)
    for sub, src in DATA_SUBDIRS.items():
        if src.exists():
            shutil.copytree(src, target / sub)
    return name


def list_snapshots() -> list:
    """Neueste zuerst."""
    if not SNAPSHOTS_DIR.exists():
        return []
    return sorted((d.name for d in SNAPSHOTS_DIR.iterdir() if d.is_dir()), reverse=True)


def restore_snapshot(name: str) -> str:
    """Ersetzt den aktuellen data/-Stand komplett durch den Snapshot-Inhalt."""
    src_root = SNAPSHOTS_DIR / name
    if not src_root.exists():
        raise FileNotFoundError(f"Snapshot '{name}' existiert nicht.")

    for sub, dst in DATA_SUBDIRS.items():
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        src = src_root / sub
        if src.exists():
            for item in src.iterdir():
                target = dst / item.name
                if item.is_file():
                    shutil.copy2(item, target)
                elif item.is_dir():
                    shutil.copytree(item, target)
    return f"Snapshot '{name}' wiederhergestellt."


def delete_snapshot(name: str) -> str:
    target = SNAPSHOTS_DIR / name
    if not target.exists():
        return f"Snapshot '{name}' existierte nicht."
    shutil.rmtree(target)
    return f"Snapshot '{name}' gelöscht."
