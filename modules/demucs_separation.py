"""
Music/Noise Removal via Demucs — läuft in-process über die Python-API
(demucs.api.Separator), NICHT per CLI-Subprocess.

Grund: Der CLI-Pfad (demucs-torchcodec) lädt Audiodateien über eine
eigene Lade-Routine, die in diesem jungen Fork (0.1.0, Stand Frühjahr
2026) bei uns leere/fehlerhafte Tensoren produzierte (AssertionError
in TensorChunk, "degrees of freedom <= 0"). Wir laden die Audiodaten
deshalb selbst zuverlässig über soundfile/librosa (dieselbe Bibliothek,
die überall sonst im Projekt funktioniert) und übergeben Demucs nur
noch den fertigen Tensor — die kaputte Datei-Lade-Routine kommt so
gar nicht mehr zum Einsatz. Das pip-Paket "demucs-torchcodec" stellt
weiterhin das gewohnte "demucs"-Python-Modul bereit (siehe Traceback:
site-packages/demucs/api.py), die Python-API bleibt also gleich.

Alternative Engine: modules/uvr_separation.py (audio-separator/UVR-Modelle),
wählbar in Tab 2 der GUI. Alles nach der reinen Trennung (Musik-Abschnitte
schneiden, Stille-Check, Denoising, Speichern) ist gemeinsamer Code in
modules/separation_common.py.
"""
import gc
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, WORKING_SAMPLE_RATE
from modules.torchaudio_compat import ensure_patched as ensure_torchaudio_compat
from modules.separation_common import (
    DemucsResult, finalize_vocals, truncate_output, list_processed_files,  # noqa: F401 (Re-Export für app.py)
    find_active_regions, build_active_only_buffer, reassemble_from_active,
)

DEMUCS_MODEL = "htdemucs_ft"
DEMUCS_TMP_DIR = PROCESSED_DIR / "_demucs_tmp"

_separator = None


def _load_separator():
    """Lazy-Load, damit die GUI startet, ohne dass sofort das Modell lädt."""
    global _separator
    if _separator is None:
        ensure_torchaudio_compat()
        from demucs.api import Separator
        _separator = Separator(model=DEMUCS_MODEL)
    return _separator


def _load_stereo_for_demucs(raw_path: Path, target_sr: int) -> torch.Tensor:
    """Lädt eine Audiodatei zuverlässig über soundfile/librosa und bringt sie
    in das von Demucs erwartete Format: float32-Tensor (Kanäle, Samples),
    stereo, bei der vom Modell erwarteten Samplerate."""
    audio, sr = sf.read(str(raw_path), always_2d=True)  # (samples, channels)
    audio = audio.T.astype(np.float32)  # (channels, samples)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, axis=-1)

    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)  # Demucs erwartet Stereo

    return torch.from_numpy(np.ascontiguousarray(audio))


def _separate_all(raw_path: Path) -> tuple:
    """Trennt ALLE Stems des Modells ab (bei htdemucs_ft: vocals, drums, bass,
    other). Gibt ({stem_name: audio_mono_float32}, samplerate, skip_fraction)
    zurück — damit können wir sowohl nur die Vocals nehmen (Modus 'Musik
    entfernen') als auch die Musik-Restenergie pro Zeitfenster berechnen
    (Modus 'Musik-Abschnitte rausschneiden').

    Vorher: Stille-Vorprüfung (find_active_regions) — nur Abschnitte mit
    tatsächlichem Pegel werden durch Demucs geschickt, längere reine Stille
    (Intro, Outro, Pausen) wird unverändert übernommen. Spart bei Dateien
    mit viel Leerlauf spürbar Rechenzeit, ohne die Zeitachse zu verändern."""
    separator = _load_separator()
    model_sr = separator.samplerate

    wav = _load_stereo_for_demucs(raw_path, model_sr)
    wav_np = wav.numpy()
    mono = wav_np.mean(axis=0)
    total_samples = wav_np.shape[1]

    regions = find_active_regions(mono, model_sr)
    skip_fraction = 1.0 - (sum(e - s for s, e in regions) / total_samples if total_samples else 0.0)

    active_only = build_active_only_buffer(wav_np, regions)
    active_tensor = torch.from_numpy(np.ascontiguousarray(active_only))
    _origin, separated_active = separator.separate_tensor(active_tensor, model_sr)

    stems = {}
    for name, tensor in separated_active.items():
        active_mono = tensor.detach().cpu().numpy().mean(axis=0).astype(np.float32)
        # In übersprungenen Stille-Abschnitten: bei "vocals" das unveränderte
        # Original einsetzen (natürlicher als digitale Nullstille), bei den
        # übrigen Stems (drums/bass/other) Stille einsetzen (dort ist ohnehin
        # nichts zu trennen).
        silent_source = mono if name == "vocals" else np.zeros_like(mono)
        stems[name] = reassemble_from_active(total_samples, regions, active_mono, silent_source, model_sr)

    return stems, model_sr, skip_fraction


def process_file(
    raw_path: Path, target_sr: int = WORKING_SAMPLE_RATE, denoise: bool = True,
    music_mode: str = "remove_music",
) -> DemucsResult:
    """music_mode:
    - 'remove_music': Musik/Rauschen aus der Sprachspur heraustrennen (Standard)
    - 'cut_music_sections': Abschnitte mit zu viel Musik-Restenergie komplett
      entfernen statt die Trennung dort zu behalten
    """
    try:
        stems, vocals_sr, skip_fraction = _separate_all(raw_path)
        vocals_audio = stems["vocals"]
        other_audio = sum(a for name, a in stems.items() if name != "vocals")

        pre_info = f"{skip_fraction * 100:.0f}% als Stille übersprungen" if skip_fraction > 0.01 else ""
        return finalize_vocals(
            raw_path, vocals_audio, other_audio, vocals_sr,
            target_sr=target_sr, denoise=denoise, music_mode=music_mode, pre_info=pre_info,
        )
    except Exception as e:
        return DemucsResult(str(raw_path), "", False, truncate_output(str(e)))


def _release_separator():
    """Modell nach dem kompletten Tab-2-Lauf (alle Dateien) aus dem GPU-
    Speicher entfernen, statt es dauerhaft resident zu halten.

    Grund: Demucs läuft im GLEICHEN Prozess wie whisperX (Tab 3) — ein
    dauerhaft gecachtes Demucs-Modell blockiert dort direkt VRAM, das
    whisperX/pyannote danach fehlt (siehe 0.20.4/0.20.5: derselbe Fehler bei
    UVR, dort über einen separaten Server gefixt; hier jetzt im Hauptprozess
    selbst). Anders als bei UVR wird hier NICHT nach jeder einzelnen Datei
    freigegeben, sondern erst nach dem kompletten Batch (process_files) —
    das Neuladen würde sich sonst bei vielen Dateien in einem Lauf unnötig
    oft wiederholen."""
    global _separator
    _separator = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def process_files(
    raw_paths: list, target_sr: int = WORKING_SAMPLE_RATE, denoise: bool = True,
    music_mode: str = "remove_music", progress_cb=None,
) -> list:
    """progress_cb(index_1_based, total, filename) wird nach jeder Datei aufgerufen,
    damit die GUI einen Fortschrittsbalken anzeigen kann."""
    results = []
    total = len(raw_paths)
    try:
        for i, p in enumerate(raw_paths, start=1):
            results.append(process_file(p, target_sr=target_sr, denoise=denoise, music_mode=music_mode))
            if progress_cb:
                progress_cb(i, total, p.name)
    finally:
        if DEMUCS_TMP_DIR.exists():
            shutil.rmtree(DEMUCS_TMP_DIR, ignore_errors=True)
        _release_separator()
    return results
