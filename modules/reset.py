"""
Reset: löscht alle Zwischenergebnisse (Raw/Processed/Segments/Export),
damit man mit einem sauberen Stand neu anfangen kann. Bleibt bestehen
über GUI-Neustarts hinweg (Daten liegen auf der Platte), deswegen
braucht es einen expliziten Reset-Button statt automatischem Löschen
beim Beenden.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, PROCESSED_DIR, SEGMENTS_DIR, EXPORT_DIR

RESET_DIRS = {
    "Import (raw)": RAW_DIR,
    "Music/Noise Removal (processed)": PROCESSED_DIR,
    "Segmentierung (segments + manifest)": SEGMENTS_DIR,
    "Export": EXPORT_DIR,
}


def reset_all() -> str:
    """Löscht den Inhalt aller Pipeline-Verzeichnisse (nicht die Verzeichnisse
    selbst). Gibt eine kurze Zusammenfassung zurück."""
    lines = []
    for label, d in RESET_DIRS.items():
        count = 0
        for item in sorted(d.rglob("*"), reverse=True):  # Dateien vor Ordnern
            if item.is_file():
                item.unlink()
                count += 1
            elif item.is_dir():
                try:
                    item.rmdir()
                except OSError:
                    pass  # nicht leer (sollte nicht vorkommen) -> überspringen
        lines.append(f"{label}: {count} Datei(en) gelöscht")
    return "\n".join(lines)
