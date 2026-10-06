"""
Laufzeit-Einstellungen für Workflow-Parameter, die im GUI-Tab "Einstellungen"
verändert werden können, ohne den Code anzufassen.

Prinzip: config.py bleibt die Quelle der WERKSEINSTELLUNGEN (Defaults) —
unverändert. Dieses Modul verwaltet zusätzlich lokale Überschreibungen, die
in settings.json (Projekt-Wurzelverzeichnis, NICHT in data/ — überlebt also
auch einen Reset) gespeichert werden. Module, die einen der hier gelisteten
Parameter benutzen, rufen settings.get("NAME") zur Laufzeit auf, statt die
config.py-Konstante direkt und fest zu importieren — eine Änderung im
Einstellungen-Tab wirkt dadurch sofort beim nächsten Durchlauf, ganz ohne
GUI-Neustart.

Bewusst nur die Parameter hier, die den WORKFLOW/das Ergebnis beeinflussen
(Clip-Länge, Schwellwerte, Trim/Padding, Lautstärke-Ziel...). Rein
technisches (Server-Ports/-URLs, Pfade, Samplerate, Modell-Dateinamen)
bleibt ausschließlich in config.py — ein Ändern davon zur Laufzeit wäre
riskant (z.B. Samplerate mitten im Projekt) oder braucht ohnehin einen
Neustart (Server-Ports).
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

SETTINGS_PATH = config.BASE_DIR / "settings.json"

# name -> (Werkseinstellung aus config.py, min, max, step, Gruppe, Nachkomma-
# stellen für die Anzeige). Gruppe steuert nur, in welchem Abschnitt des
# Einstellungen-Tabs ein Parameter erscheint (siehe app.py) — keine
# funktionale Bedeutung hier.
SETTINGS_SPEC = {
    # --- Segmentierung (Tab 3) ---
    "MIN_CLIP_DURATION_S": (config.MIN_CLIP_DURATION_S, 0.5, 10.0, 0.1, "segmentation", 1),
    "MAX_CLIP_DURATION_S": (config.MAX_CLIP_DURATION_S, 2.0, 20.0, 0.5, "segmentation", 1),
    "MIN_SILENCE_GAP_S": (config.MIN_SILENCE_GAP_S, 0.05, 2.0, 0.05, "segmentation", 2),
    "SILERO_VAD_THRESHOLD": (config.SILERO_VAD_THRESHOLD, 0.1, 0.9, 0.05, "segmentation", 2),
    "WORD_BOUNDARY_PAD_S": (config.WORD_BOUNDARY_PAD_S, 0.0, 0.5, 0.01, "segmentation", 2),
    "ENERGY_SNAP_RADIUS_S": (config.ENERGY_SNAP_RADIUS_S, 0.0, 0.5, 0.01, "segmentation", 2),
    "SILENCE_TRIM_TOP_DB": (config.SILENCE_TRIM_TOP_DB, 10, 60, 1, "segmentation", 0),
    "SILENCE_TRIM_MAX_S": (config.SILENCE_TRIM_MAX_S, 0.0, 0.5, 0.01, "segmentation", 2),
    "FADE_DURATION_S": (config.FADE_DURATION_S, 0.0, 0.5, 0.01, "segmentation", 2),
    "WHISPERX_BATCH_SIZE": (config.WHISPERX_BATCH_SIZE, 1, 32, 1, "segmentation", 0),
    "RECLIP_MARGIN_S": (config.RECLIP_MARGIN_S, 0.5, 5.0, 0.5, "segmentation", 1),
    "RECLIP_MAX_ATTEMPTS": (config.RECLIP_MAX_ATTEMPTS, 1, 5, 1, "segmentation", 0),
    # --- Music/Noise Removal (Tab 2) ---
    "MUSIC_CUT_WINDOW_S": (config.MUSIC_CUT_WINDOW_S, 0.2, 5.0, 0.1, "music", 1),
    "MUSIC_CUT_ENERGY_RATIO": (config.MUSIC_CUT_ENERGY_RATIO, 0.05, 0.9, 0.05, "music", 2),
    "SILENCE_SKIP_WINDOW_S": (config.SILENCE_SKIP_WINDOW_S, 0.1, 2.0, 0.1, "music", 1),
    "SILENCE_SKIP_MIN_DURATION_S": (config.SILENCE_SKIP_MIN_DURATION_S, 0.5, 10.0, 0.5, "music", 1),
    "SILENCE_SKIP_RMS_THRESHOLD": (config.SILENCE_SKIP_RMS_THRESHOLD, 0.0001, 0.02, 0.0005, "music", 4),
    "SILENCE_SKIP_CROSSFADE_S": (config.SILENCE_SKIP_CROSSFADE_S, 0.0, 0.3, 0.01, "music", 2),
    # --- Qualitäts-Schwellwerte (Tab 4) ---
    "NISQA_MOS_RED_THRESHOLD": (config.NISQA_MOS_RED_THRESHOLD, 1.0, 5.0, 0.1, "quality", 1),
    "ASR_CONFIDENCE_RED_THRESHOLD": (config.ASR_CONFIDENCE_RED_THRESHOLD, 0.0, 1.0, 0.05, "quality", 2),
    # --- Export (Tab 5) ---
    "TARGET_LUFS": (config.TARGET_LUFS, -36.0, -10.0, 0.5, "export", 1),
}

_overrides: dict = {}


def _load():
    global _overrides
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            _overrides = {k: v for k, v in data.items() if k in SETTINGS_SPEC}
        except Exception as e:
            print(f"[settings] settings.json konnte nicht gelesen werden ({e}) — nutze Werkseinstellungen.")
            _overrides = {}
    else:
        _overrides = {}


_load()


def get(name: str):
    """Aktueller Wert: eigene Einstellung falls vorhanden, sonst Werks-
    einstellung aus config.py."""
    if name not in SETTINGS_SPEC:
        raise KeyError(f"Unbekannter Einstellungs-Parameter: {name}")
    return _overrides.get(name, SETTINGS_SPEC[name][0])


def get_all() -> dict:
    return {name: get(name) for name in SETTINGS_SPEC}


def get_defaults() -> dict:
    return {name: spec[0] for name, spec in SETTINGS_SPEC.items()}


def set_values(values: dict):
    """Setzt mehrere Werte auf einmal und speichert sofort nach settings.json.
    Werte, die exakt der Werkseinstellung entsprechen, werden NICHT als
    Überschreibung gespeichert — hält settings.json klein und macht auf
    einen Blick sichtbar, was tatsächlich verändert wurde."""
    global _overrides
    for name, value in values.items():
        if name not in SETTINGS_SPEC:
            continue
        default = SETTINGS_SPEC[name][0]
        if value == default:
            _overrides.pop(name, None)
        else:
            _overrides[name] = value
    _save()


def reset_all():
    global _overrides
    _overrides = {}
    _save()


def _save():
    with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
        json.dump(_overrides, fh, ensure_ascii=False, indent=2)
