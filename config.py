"""
Zentrale Konfiguration für das Dataset-Prep-Tool.
Alle Module importieren Pfade und Standardwerte von hier, statt sie
hart zu codieren, damit das Tool an eine andere Verzeichnisstruktur
angepasst werden kann, ohne mehrere Dateien anzufassen.
"""
from pathlib import Path

# --- Projekt-Version ---
# Semantisch: <1.0.0 solange kein vollständiger Echtdurchlauf (alle 5 Tabs,
# echtes Audio) bestätigt wurde. Wird in Snapshot-Namen mit eingebettet,
# siehe modules/snapshot.py, und im GUI-Header angezeigt.
PROJECT_VERSION = "0.13.3"

# --- Verzeichnisstruktur ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"                # Original-Audio, so wie importiert
PROCESSED_DIR = DATA_DIR / "processed"    # nach Demucs (Musik/Noise entfernt)
SEGMENTS_DIR = DATA_DIR / "segments"      # einzelne Clips nach VAD-Segmentierung
EXPORT_DIR = DATA_DIR / "export"          # fertige Datasets (XTTS/RVC)

for d in (RAW_DIR, PROCESSED_DIR, SEGMENTS_DIR, EXPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Audio-Standards ---
# Interner Arbeitsstandard, bis feststeht, welche der beiden Ziel-WebUIs
# selbst resampled. 22050 Hz ist der übliche XTTS-v2-Trainingsstandard.
WORKING_SAMPLE_RATE = 22050

EXPORT_SAMPLE_RATES = {
    "xtts": 22050,
    "rvc": 40000,   # RVC-Trainingsstandard; ggf. auf 48000 anpassen je nach RVC-Variante
}

# --- Lautstärke-Normalisierung (beim Export) ---
TARGET_LUFS = -23.0  # Broadcast-/HUI-Audio-Corpus-Standard, einheitlich für konsistentes Training
PEAK_LIMIT = 0.99     # verhindert Clipping nach der Lautstärke-Anhebung

# --- Qualitäts-Scoring (Review-Tab) ---
NISQA_SERVER_URL = "http://localhost:8050/predict"  # separater NISQA-Server, siehe nisqa_server/
NISQA_MOS_RED_THRESHOLD = 2.5   # NISQA-Skala 1-5: darunter = rot markiert
ASR_CONFIDENCE_RED_THRESHOLD = 0.6  # whisperX-Wortkonfidenz 0-1: darunter = rot markiert

# --- Musik-Behandlung (Tab 2) ---
# Alternative zur reinen Trennung: Abschnitte mit zu viel Musik-Restenergie
# komplett herausschneiden statt die (evtl. unsaubere) Trennung zu behalten.
MUSIC_CUT_WINDOW_S = 1.0          # Analyse-Fenstergröße in Sekunden
MUSIC_CUT_ENERGY_RATIO = 0.3      # Anteil Musik-Energie an Gesamtenergie, ab dem ein Fenster rausfliegt

# --- VAD / Segmentierung ---
# Zielwerte für Clip-Länge (Sekunden) — orientiert an XTTS/RVC-Trainingsempfehlungen
MIN_CLIP_DURATION_S = 2.0
MAX_CLIP_DURATION_S = 15.0
# Mindest-Stille (Sekunden) zwischen zwei Segmenten, um Atmer/Breath-Reste
# nicht als eigenes Clip-Ende zu werten
MIN_SILENCE_GAP_S = 0.3

SILERO_VAD_THRESHOLD = 0.5  # Sprach-Wahrscheinlichkeit, ab der als Sprache gewertet wird

# --- Clip-Nachbearbeitung ---
SILENCE_TRIM_TOP_DB = 35     # librosa.effects.trim: alles leiser als (max_dB - top_db) wird an den Rändern abgeschnitten
FADE_DURATION_S = 0.1        # Fade-in/-out an jedem Clip-Rand, gegen Klick-/Atem-Reste

# --- Transkription ---
WHISPERX_MODEL = "large-v3"
# WHISPERX_LANGUAGE ist nur noch der Default-Wert für das Sprach-Dropdown in
# der GUI (Tab 3) — die tatsächlich verwendete Sprache kommt zur Laufzeit aus
# der Nutzerauswahl, siehe app.py.
WHISPERX_LANGUAGE = "de"

# Sprachen mit dediziertem whisperX-Alignment-Modell (Wort-Timestamps).
# Format: (Code, deutsches Label, englisches Label). Deutsch zuerst = Default.
SUPPORTED_TRANSCRIPTION_LANGUAGES = [
    ("de", "Deutsch", "German"),
    ("en", "Englisch", "English"),
    ("fr", "Französisch", "French"),
    ("es", "Spanisch", "Spanish"),
    ("it", "Italienisch", "Italian"),
    ("pt", "Portugiesisch", "Portuguese"),
    ("nl", "Niederländisch", "Dutch"),
    ("pl", "Polnisch", "Polish"),
    ("ru", "Russisch", "Russian"),
    ("uk", "Ukrainisch", "Ukrainian"),
    ("cs", "Tschechisch", "Czech"),
    ("sk", "Slowakisch", "Slovak"),
    ("hu", "Ungarisch", "Hungarian"),
    ("ro", "Rumänisch", "Romanian"),
    ("hr", "Kroatisch", "Croatian"),
    ("sl", "Slowenisch", "Slovenian"),
    ("el", "Griechisch", "Greek"),
    ("tr", "Türkisch", "Turkish"),
    ("da", "Dänisch", "Danish"),
    ("fi", "Finnisch", "Finnish"),
    ("no", "Norwegisch", "Norwegian"),
    ("he", "Hebräisch", "Hebrew"),
    ("ar", "Arabisch", "Arabic"),
    ("fa", "Persisch", "Persian"),
    ("vi", "Vietnamesisch", "Vietnamese"),
    ("ko", "Koreanisch", "Korean"),
    ("ja", "Japanisch", "Japanese"),
    ("zh", "Chinesisch", "Chinese"),
    ("hi", "Hindi", "Hindi"),
    ("ur", "Urdu", "Urdu"),
    ("ca", "Katalanisch", "Catalan"),
    ("gl", "Galizisch", "Galician"),
    ("eu", "Baskisch", "Basque"),
    ("ka", "Georgisch", "Georgian"),
    ("lv", "Lettisch", "Latvian"),
    ("tl", "Tagalog", "Tagalog"),
]
WHISPERX_COMPUTE_TYPE = "float16"  # bei RTX 2080 Ti (CUDA) sinnvoll; auf "int8" fallback für CPU

# --- Export ---
LJSPEECH_METADATA_FILENAME = "metadata.csv"  # Format: wav_filename|transcription
