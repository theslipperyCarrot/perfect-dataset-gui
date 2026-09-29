"""
Export der freigegebenen Clips in beide Zielformate:

  - XTTS: LJSpeech-Struktur
        data/export/xtts/wavs/<clip_id>.wav   (22050 Hz, Mono)
        data/export/xtts/metadata.csv         (clip_id|text|text)
  - RVC: reine Audio-Sammlung, kein Text nötig (RVC-Training ist
    unsupervised bzgl. Transkript)
        data/export/rvc/wavs/<clip_id>.wav    (40000 Hz, Mono)

Nur Clips mit approved == True werden exportiert.
"""
import csv
import sys
from pathlib import Path
from dataclasses import dataclass

import soundfile as sf
import librosa
import numpy as np
import pyloudnorm as pyln

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SEGMENTS_DIR, EXPORT_DIR, EXPORT_SAMPLE_RATES, LJSPEECH_METADATA_FILENAME, PEAK_LIMIT
from modules import settings
from modules.segment_and_transcribe import ClipEntry, load_manifest

XTTS_DIR = EXPORT_DIR / "xtts"
XTTS_WAVS_DIR = XTTS_DIR / "wavs"
RVC_DIR = EXPORT_DIR / "rvc"
RVC_WAVS_DIR = RVC_DIR / "wavs"


@dataclass
class ExportSummary:
    xtts_count: int = 0
    rvc_count: int = 0
    skipped_no_text: int = 0
    errors: list = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def _normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float = None) -> np.ndarray:
    """Normalisiert auf eine einheitliche LUFS-Lautstärke, mit Peak-Limiting
    gegen Clipping. Bei (quasi) Stille ist integrated_loudness -inf — in dem
    Fall bleibt der Clip unverändert, statt mit einem Fantasie-Gain
    hochgerechnet zu werden. target_lufs: None -> aktuelle Einstellung aus
    dem Einstellungen-Tab."""
    if target_lufs is None:
        target_lufs = settings.get("TARGET_LUFS")
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(audio)
    except Exception:
        return audio  # z.B. Clip zu kurz für die LUFS-Messung -> unverändert lassen
    if not np.isfinite(loudness):
        return audio

    normalized = pyln.normalize.loudness(audio, loudness, target_lufs)

    peak = np.max(np.abs(normalized))
    if peak > PEAK_LIMIT:
        normalized = normalized * (PEAK_LIMIT / peak)

    return normalized


def _resample_and_write(source_path: Path, target_path: Path, target_sr: int):
    audio, sr = sf.read(str(source_path))
    if sr != target_sr:
        audio = librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    audio = _normalize_loudness(audio.astype("float32"), sr)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target_path), audio, target_sr, subtype="PCM_16")


def _approved_entries() -> list[ClipEntry]:
    """Alles, was noch im Manifest steht, gilt als freigegeben — Löschen im
    Review-Tab entfernt einen Clip direkt aus dem Manifest (siehe app.py)."""
    return load_manifest()


def export_xtts(entries: list[ClipEntry]) -> tuple[int, int, list[str]]:
    """Gibt (exportiert, ohne_text_übersprungen, fehler) zurück."""
    XTTS_WAVS_DIR.mkdir(parents=True, exist_ok=True)
    target_sr = EXPORT_SAMPLE_RATES["xtts"]

    rows = []
    skipped = 0
    errors = []
    for e in entries:
        if not e.text or not e.text.strip():
            skipped += 1
            continue
        source_path = SEGMENTS_DIR / e.clip_filename
        target_path = XTTS_WAVS_DIR / f"{e.id}.wav"
        try:
            _resample_and_write(source_path, target_path, target_sr)
            rows.append([e.id, e.text.strip(), e.text.strip()])
        except Exception as ex:
            errors.append(f"{e.id}: {ex}")

    metadata_path = XTTS_DIR / LJSPEECH_METADATA_FILENAME
    with open(metadata_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|", lineterminator="\n")
        writer.writerows(rows)

    return len(rows), skipped, errors


def export_rvc(entries: list[ClipEntry]) -> tuple[int, list[str]]:
    RVC_WAVS_DIR.mkdir(parents=True, exist_ok=True)
    target_sr = EXPORT_SAMPLE_RATES["rvc"]

    count = 0
    errors = []
    for e in entries:
        source_path = SEGMENTS_DIR / e.clip_filename
        target_path = RVC_WAVS_DIR / f"{e.id}.wav"
        try:
            _resample_and_write(source_path, target_path, target_sr)
            count += 1
        except Exception as ex:
            errors.append(f"{e.id}: {ex}")

    return count, errors


def export_all() -> ExportSummary:
    entries = _approved_entries()
    xtts_count, skipped, xtts_errors = export_xtts(entries)
    rvc_count, rvc_errors = export_rvc(entries)
    return ExportSummary(
        xtts_count=xtts_count,
        rvc_count=rvc_count,
        skipped_no_text=skipped,
        errors=xtts_errors + rvc_errors,
    )
