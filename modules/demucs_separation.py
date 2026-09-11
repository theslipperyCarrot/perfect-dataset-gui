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
"""
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import librosa
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, WORKING_SAMPLE_RATE, MUSIC_CUT_WINDOW_S, MUSIC_CUT_ENERGY_RATIO
from modules.torchaudio_compat import ensure_patched as ensure_torchaudio_compat

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


def _truncate_output(text: str, head: int = 800, tail: int = 1200) -> str:
    """Behält Anfang UND Ende einer langen Fehlerausgabe, statt nur das Ende —
    wichtige frühe Meldungen (z.B. fehlende Abhängigkeiten) gehen sonst verloren."""
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n[... gekürzt ...]\n{text[-tail:]}"


@dataclass
class DemucsResult:
    source_path: str
    target_path: str
    success: bool
    message: str = ""
    info: str = ""  # zusätzliche Erfolgs-Info (z.B. "12% wegen Musik herausgeschnitten")


def _separate_all(raw_path: Path) -> tuple:
    """Trennt ALLE Stems des Modells ab (bei htdemucs_ft: vocals, drums, bass,
    other). Gibt ({stem_name: audio_mono_float32}, samplerate) zurück — damit
    können wir sowohl nur die Vocals nehmen (Modus 'Musik entfernen') als auch
    die Musik-Restenergie pro Zeitfenster berechnen (Modus 'Musik-Abschnitte
    rausschneiden')."""
    separator = _load_separator()
    model_sr = separator.samplerate

    wav = _load_stereo_for_demucs(raw_path, model_sr)
    _origin, separated = separator.separate_tensor(wav, model_sr)
    stems = {
        name: tensor.detach().cpu().numpy().mean(axis=0).astype(np.float32)
        for name, tensor in separated.items()
    }
    return stems, model_sr


SILENCE_RMS_THRESHOLD = 1e-4  # ~ -80 dBFS; deutlich unter jeder normalen Sprachaufnahme


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0


def _cut_music_sections(vocals: np.ndarray, music: np.ndarray, sr: int) -> tuple:
    """Schneidet Zeitfenster heraus, in denen die Musik-Restenergie einen zu
    hohen Anteil an der Gesamtenergie hat, statt die (evtl. unsaubere)
    Trennung dort zu behalten. Gibt (gekürztes_audio, entfernter_anteil) zurück."""
    window = max(1, int(MUSIC_CUT_WINDOW_S * sr))
    n = len(vocals)
    kept_chunks = []
    cut_samples = 0
    for start in range(0, n, window):
        end = min(n, start + window)
        v_rms = _rms(vocals[start:end])
        m_rms = _rms(music[start:end])
        ratio = m_rms / (v_rms + m_rms + 1e-9)
        if ratio > MUSIC_CUT_ENERGY_RATIO:
            cut_samples += (end - start)
            continue
        kept_chunks.append(vocals[start:end])
    result = np.concatenate(kept_chunks) if kept_chunks else np.array([], dtype=np.float32)
    cut_fraction = (cut_samples / n) if n else 0.0
    return result, cut_fraction


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
        stems, vocals_sr = _separate_all(raw_path)
        vocals_audio = stems["vocals"]
        info = ""

        if music_mode == "cut_music_sections":
            music_audio = sum(a for name, a in stems.items() if name != "vocals")
            vocals_audio, cut_fraction = _cut_music_sections(vocals_audio, music_audio, vocals_sr)
            info = f"{cut_fraction * 100:.0f}% wegen Musik herausgeschnitten"
            if len(vocals_audio) == 0:
                return DemucsResult(
                    str(raw_path), "", False,
                    "Nach dem Herausschneiden der Musik-Abschnitte blieb nichts übrig — "
                    "vermutlich ist die ganze Datei musikdominiert.",
                )

        rms_after_demucs = _rms(vocals_audio)
        if rms_after_demucs < SILENCE_RMS_THRESHOLD:
            return DemucsResult(
                str(raw_path), "", False,
                f"Ergebnis nach Demucs ist verdächtig leise/leer (RMS={rms_after_demucs:.2e}) — "
                "vermutlich ein Fehler in der Verarbeitung statt eines echten Ergebnisses, "
                "kein 'Erfolg' zum Sicherheitscheck. Bitte Ausgangsdatei/Terminal-Log prüfen.",
            )

        if denoise:
            DEMUCS_TMP_DIR.mkdir(parents=True, exist_ok=True)
            tmp_in = DEMUCS_TMP_DIR / f"{raw_path.stem}_vocals.wav"
            sf.write(str(tmp_in), vocals_audio, vocals_sr, subtype="PCM_16")

            from modules.denoise import denoise_file
            denoised_path = DEMUCS_TMP_DIR / f"{raw_path.stem}_denoised.wav"
            denoise_file(tmp_in, denoised_path)
            vocals_audio, vocals_sr = sf.read(str(denoised_path))

            rms_after_denoise = _rms(np.asarray(vocals_audio))
            if rms_after_denoise < SILENCE_RMS_THRESHOLD:
                return DemucsResult(
                    str(raw_path), "", False,
                    f"Ergebnis nach DeepFilterNet-Denoising ist verdächtig leise/leer "
                    f"(RMS={rms_after_denoise:.2e}) — Demucs-Ausgabe war noch ok, das "
                    "Denoising hat das Signal aber offenbar komplett weggefiltert.",
                )

        if vocals_sr != target_sr:
            vocals_audio = librosa.resample(
                np.asarray(vocals_audio, dtype=np.float32), orig_sr=vocals_sr, target_sr=target_sr,
            )

        target_path = PROCESSED_DIR / f"{raw_path.stem}.wav"
        sf.write(str(target_path), vocals_audio, target_sr, subtype="PCM_16")

        return DemucsResult(str(raw_path), str(target_path), True, info=info)
    except Exception as e:
        return DemucsResult(str(raw_path), "", False, _truncate_output(str(e)))


def process_files(
    raw_paths: list, target_sr: int = WORKING_SAMPLE_RATE, denoise: bool = True,
    music_mode: str = "remove_music", progress_cb=None,
) -> list:
    """progress_cb(index_1_based, total, filename) wird nach jeder Datei aufgerufen,
    damit die GUI einen Fortschrittsbalken anzeigen kann."""
    results = []
    total = len(raw_paths)
    for i, p in enumerate(raw_paths, start=1):
        results.append(process_file(p, target_sr=target_sr, denoise=denoise, music_mode=music_mode))
        if progress_cb:
            progress_cb(i, total, p.name)
    if DEMUCS_TMP_DIR.exists():
        shutil.rmtree(DEMUCS_TMP_DIR, ignore_errors=True)
    return results


def list_processed_files() -> list:
    return sorted(PROCESSED_DIR.glob("*.wav"))
