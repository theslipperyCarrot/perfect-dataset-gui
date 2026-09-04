"""
Import & Vorverarbeitung: Audio laden, auf einheitliche Samplerate/Kanäle
bringen und im RAW-Verzeichnis für die weiteren Pipeline-Schritte ablegen.
"""
from pathlib import Path
from dataclasses import dataclass

import soundfile as sf
import librosa
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW_DIR, WORKING_SAMPLE_RATE


@dataclass
class ImportResult:
    source_path: str
    target_path: str
    duration_s: float
    original_sr: int
    target_sr: int


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Mehrkanal-Audio auf Mono reduzieren (Mittelwert über Kanäle)."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def import_audio_file(source_path: str, target_sr: int = WORKING_SAMPLE_RATE) -> ImportResult:
    """
    Lädt eine einzelne Audiodatei, konvertiert auf Mono + Ziel-Samplerate
    und speichert sie als WAV im RAW-Verzeichnis.

    Bewusst konservativ: keine Lautstärke-Normalisierung an dieser Stelle,
    das passiert erst nach der Demucs-Trennung, damit Normalisierung nicht
    durch Musik-/Rauschanteile verzerrt wird.
    """
    source = Path(source_path)
    audio, orig_sr = librosa.load(str(source), sr=None, mono=False)
    audio = _to_mono(audio) if audio.ndim > 1 else audio

    if orig_sr != target_sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    target_path = RAW_DIR / f"{source.stem}.wav"
    sf.write(str(target_path), audio, target_sr, subtype="PCM_16")

    duration_s = len(audio) / target_sr
    return ImportResult(
        source_path=str(source),
        target_path=str(target_path),
        duration_s=duration_s,
        original_sr=orig_sr,
        target_sr=target_sr,
    )


def import_audio_files(source_paths: list[str], target_sr: int = WORKING_SAMPLE_RATE) -> list[ImportResult]:
    """Batch-Import für mehrere Dateien (z.B. aus dem Gradio File-Upload)."""
    results = []
    for p in source_paths:
        try:
            results.append(import_audio_file(p, target_sr=target_sr))
        except Exception as e:
            results.append(ImportResult(
                source_path=p, target_path="", duration_s=0.0,
                original_sr=0, target_sr=target_sr,
            ))
            print(f"[audio_io] Fehler beim Import von {p}: {e}")
    return results


def list_raw_files() -> list[Path]:
    return sorted(RAW_DIR.glob("*.wav"))
