"""
Zentrale Konfiguration für das Dataset-Prep-Tool.
Alle Module importieren Pfade und Standardwerte von hier, statt sie
hart zu codieren, damit das Tool an eine andere Verzeichnisstruktur
angepasst werden kann, ohne mehrere Dateien anzufassen.

Die hier eingetragenen Werte sind die WERKSEINSTELLUNGEN. Ein Teil davon
(alles, was den Workflow/das Ergebnis beeinflusst — Clip-Länge, Trim/
Padding, Schwellwerte, Lautstärke-Ziel etc.) kann zusätzlich zur Laufzeit
im GUI-Tab "Einstellungen" überschrieben werden, siehe modules/settings.py
und settings.json (Projekt-Wurzelverzeichnis). Diese Datei hier bleibt der
Fallback/die Werkseinstellung — ein Ändern eines Wertes hier setzt also nur
den Default zurück, eine bestehende Überschreibung in settings.json bleibt
davon unberührt. Rein Technisches (Server-Ports/-URLs, Pfade, Samplerate,
Modell-Dateinamen) ist NICHT über die GUI änderbar, nur hier.
"""
from pathlib import Path

# --- Projekt-Version ---
# Semantisch: <1.0.0 solange kein vollständiger Echtdurchlauf (alle 5 Tabs,
# echtes Audio) bestätigt wurde. Wird in Snapshot-Namen mit eingebettet,
# siehe modules/snapshot.py, und im GUI-Header angezeigt.
PROJECT_VERSION = "0.23.2"

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
DENOISE_SERVER_URL = "http://localhost:8051/denoise"  # separater Denoise-Server, siehe denoise_server/
SEPARATOR_SERVER_URL = "http://localhost:8052/separate"  # separater UVR-Server, siehe separator_server/
# Modell für die UVR-Engine (audio-separator). BS-Roformer gilt aktuell als
# eines der stärksten frei verfügbaren Vocal-Isolation-Modelle. Alternativen
# (z.B. "UVR-MDX-NET-Inst_HQ_3.onnx", "Kim_Vocal_2.onnx") können hier
# eingetragen werden; volle Liste: separator_server/.venv/bin/python -c
# "from audio_separator.separator import Separator; print(Separator().list_supported_model_files())"
SEPARATOR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
NISQA_MOS_RED_THRESHOLD = 2.5   # NISQA-Skala 1-5: darunter = rot markiert
ASR_CONFIDENCE_RED_THRESHOLD = 0.6  # whisperX-Wortkonfidenz 0-1: darunter = rot markiert

# --- Musik-Behandlung (Tab 2) ---
# Alternative zur reinen Trennung: Abschnitte mit zu viel Musik-Restenergie
# komplett herausschneiden statt die (evtl. unsaubere) Trennung zu behalten.
MUSIC_CUT_WINDOW_S = 1.0          # Analyse-Fenstergröße in Sekunden
MUSIC_CUT_ENERGY_RATIO = 0.3      # Anteil Musik-Energie an Gesamtenergie, ab dem ein Fenster rausfliegt

# --- Stille-Vorprüfung vor Music/Noise Removal (Tab 2) ---
# Läuft VOR der eigentlichen (teuren) Trennung: reine, längere Stille-
# Abschnitte (Intro, Outro, lange Pausen) werden erkannt und NICHT durch
# Demucs/UVR geschickt, sondern unverändert übernommen — spart bei Dateien
# mit viel Leerlauf spürbar Zeit. Bewusst nur ein Pegel-Check (kein VAD):
# VAD ist ausgerechnet bei lauter Hintergrundmusik unzuverlässig, genau dort
# wo die Trennung am nötigsten ist. Siehe modules/separation_common.py,
# find_active_regions().
SILENCE_SKIP_WINDOW_S = 0.5        # Analyse-Fenstergröße für den Pegel-Check
SILENCE_SKIP_MIN_DURATION_S = 2.0  # ab dieser Stille-Länge wird übersprungen (kürzere Pausen bleiben "aktiv")
SILENCE_SKIP_RMS_THRESHOLD = 2e-3  # deutlich unter Sprache/Musik, aber über normalem Rauschboden
SILENCE_SKIP_CROSSFADE_S = 0.03    # Crossfade an jeder Naht zwischen übersprungener Stille und Trennungs-Ergebnis

# --- VAD / Segmentierung ---
# Zielwerte für Clip-Länge (Sekunden) — orientiert an XTTS/RVC-Trainingsempfehlungen
MIN_CLIP_DURATION_S = 2.0
MAX_CLIP_DURATION_S = 10.0
# Mindest-Stille (Sekunden) zwischen zwei Segmenten, um Atmer/Breath-Reste
# nicht als eigenes Clip-Ende zu werten
MIN_SILENCE_GAP_S = 0.3

SILERO_VAD_THRESHOLD = 0.5  # Sprach-Wahrscheinlichkeit, ab der als Sprache gewertet wird

# whisperX-Wortausrichtung ist am Rand nicht perfekt (Wortanfang/-ende kann
# leicht daneben liegen) — jeder Clip bekommt deshalb vor dem ersten und nach
# dem letzten Wort etwas Vorlauf/Nachlauf dazu, geclippt an den Nachbar-Clip
# (nie Überlappung, kein doppeltes Audiomaterial). Schutz gegen das
# "fehlendes erstes Wort"-Problem, falls der gemeldete Wort-Timestamp zu spät
# ansetzt. Siehe _pad_clip_boundaries().
WORD_BOUNDARY_PAD_S = 0.08

# Zusätzlich zum Wortrand-Polster: jede Clip-Grenze (nach dem Polster) wird
# in einem kleinen Radius auf den Punkt der geringsten lokalen Energie im
# tatsächlichen Audiosignal "gesnappt" — korrigiert Millisekunden-
# Ungenauigkeit in whisperX' Wort-Zeitstempeln, die auch mit mehr Kontext
# (Kontrollschleife) bestehen bleibt, weil sie nicht an fehlendem Kontext
# liegt, sondern an der Ausrichtung selbst. 0 deaktiviert das Snapping.
# Siehe _snap_to_energy_minimum().
ENERGY_SNAP_RADIUS_S = 0.15

# --- Clip-Nachbearbeitung ---
SILENCE_TRIM_TOP_DB = 35     # librosa.effects.trim: alles leiser als (max_dB - top_db) wird an den Rändern abgeschnitten
# Obergrenze, wie viel der Rand-Trim pro Seite maximal wegschneiden darf.
# Schutz gegen leise Rand-Wörter (z.B. ein unbetontes "Sie"), die relativ zum
# lautesten Punkt im Clip fälschlich als Stille erkannt und sonst komplett
# mit weggeschnitten würden. Jetzt klein gehalten (statt alleinigem Schutz),
# weil WORD_BOUNDARY_PAD_S oben bereits gezielt Sicherheitsabstand vor dem
# ersten/nach dem letzten Wort schafft — der Trim muss nur noch das kleine
# bisschen Polster glätten, nicht mehr potenziell echten Wortinhalt retten.
SILENCE_TRIM_MAX_S = 0.06
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
# Wie viele Audio-Chunks whisperX gleichzeitig verarbeitet. Höher = schneller,
# aber proportional mehr VRAM. whisperX' eigener Default (32) ist zu hoch,
# wenn parallel noch andere Modelle/Server (Demucs/UVR/Denoise) auf derselben
# GPU laufen -> "CUDA out of memory" bzw. "batch_size probably too large".
# Einstellbar im Einstellungen-Tab.
WHISPERX_BATCH_SIZE = 8

# --- Round-Trip-Check (Tab 4) ---
# Unabhängige zweite Transkription jedes fertigen Clips (per faster-whisper,
# OHNE Wort-Alignment — wir brauchen hier nur den reinen Text, kein Timing)
# zum Abgleich mit dem zugewiesenen Text. Deckt Fehler auf, die die
# eigentliche Segmentierung durchrutschen lässt (z.B. ein mit ins Audio
# gerutschtes Nachbarwort) — ein isoliert transkribierter Clip enthält dann
# ein Wort zu viel/zu wenig, was der Wort-Fehlerrate (WER) deutlich stärker
# auffällt als ein reiner Alignment-Konfidenzwert. Bewusst ein kleineres
# Modell als WHISPERX_MODEL (large-v3): hier zählt Tempo über viele Clips
# hinweg mehr als letzte Genauigkeit, und Fehler durch das kleinere Modell
# selbst wirken sich nur auf den EINEN Vergleichswert aus, nicht auf die
# eigentliche Transkription.
ROUNDTRIP_MODEL = "medium"
ROUNDTRIP_COMPUTE_TYPE = "float16"

# --- Automatisches Nachschneiden (Kontrollschleife, Tab 4) ---
# Für Clips unterhalb der Round-Trip-Warnschwelle: erweitertes Zeitfenster
# aus der Quelldatei, neu transkribieren+ausrichten, beste Wortfolge zum
# erwarteten Text suchen. Schlägt fehl (oder bringt keine Verbesserung),
# bleibt der Original-Clip unangetastet. RECLIP_MARGIN_S ist die Fenster-
# Erweiterung beim ERSTEN Versuch; jeder weitere Versuch verdoppelt sie
# (Versuch 2: 2×, Versuch 3: 3×, ...) bis RECLIP_MAX_ATTEMPTS erreicht ist.
RECLIP_MARGIN_S = 1.5
RECLIP_MAX_ATTEMPTS = 2

# --- Export ---
LJSPEECH_METADATA_FILENAME = "metadata.csv"  # Format: wav_filename|transcription
