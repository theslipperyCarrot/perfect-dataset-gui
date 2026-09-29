"""
Music/Noise Removal über "audio-separator" (bündelt UVR-Modelle: MDX-Net,
VR-Arch, BS-Roformer). Alternative zu Demucs, wählbar in Tab 2 der GUI.

Läuft als SEPARATER Server (separator_server/, eigene venv), gleiches Muster
wie nisqa_server/ und denoise_server/: audio-separator braucht numpy>=2 und
eine eigene onnxruntime/Modell-Download-Kette, die nicht zwingend mit den
übrigen Projekt-Abhängigkeiten in derselben Umgebung koexistieren muss.
Client hier ruft den Server nur per HTTP auf.

Alles nach der reinen Trennung (Musik-Abschnitte schneiden, Stille-Check,
Denoising, Speichern) ist gemeinsamer Code in modules/separation_common.py
— identisch zur Demucs-Engine, nur die eigentliche Trennung unterscheidet
sich.
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import librosa

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, WORKING_SAMPLE_RATE, SEPARATOR_SERVER_URL, SEPARATOR_MODEL
from modules.separation_common import (
    DemucsResult, finalize_vocals, truncate_output, list_processed_files,  # noqa: F401 (Re-Export für app.py)
    find_active_regions, build_active_only_buffer, reassemble_from_active,
)


def _separate_all(raw_path: Path, timeout: float = 600.0) -> tuple:
    """Stille-Vorprüfung + Schickt NUR die aktiven Abschnitte an den
    separator_server (als ein zusammenhängender Puffer, ein einziger
    HTTP-Request), bekommt ein ZIP mit vocals.wav + instrumental.wav für
    diesen (kürzeren) Puffer zurück und baut das Ergebnis dann wieder auf
    die Original-Zeitachse zurück. Timeout bewusst hoch (600s) — UVR-Modelle
    (v.a. BS-Roformer) sind deutlich rechenintensiver als Demucs, gerade beim
    ersten Aufruf inkl. Modell-Download.

    Gibt (vocals_audio, other_audio, sr, skip_fraction) zurück."""
    orig_audio, orig_sr = sf.read(str(raw_path), always_2d=True)
    mono = orig_audio.mean(axis=1).astype(np.float32)
    total_samples = len(mono)

    regions = find_active_regions(mono, orig_sr)
    skip_fraction = 1.0 - (sum(e - s for s, e in regions) / total_samples if total_samples else 0.0)
    active_only = build_active_only_buffer(mono, regions)

    buf = io.BytesIO()
    sf.write(buf, active_only, orig_sr, format="WAV", subtype="PCM_16")
    buf.seek(0)

    resp = requests.post(
        SEPARATOR_SERVER_URL,
        files={"file": ("input.wav", buf, "audio/wav")},
        data={"model": SEPARATOR_MODEL},
        timeout=timeout,
    )
    resp.raise_for_status()

    vocals_active = None
    other_active = None
    result_sr = None
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            audio, file_sr = sf.read(io.BytesIO(zf.read(name)), always_2d=True)
            audio = audio.mean(axis=1).astype(np.float32)  # Mono
            result_sr = file_sr
            if "vocals" in name.lower():
                vocals_active = audio
            else:
                other_active = audio

    if vocals_active is None:
        raise RuntimeError(
            "separator_server hat kein 'vocals'-Stem zurückgegeben "
            "(unerwarteter Dateiname im ZIP-Ergebnis)."
        )
    if other_active is None:
        other_active = np.zeros_like(vocals_active)

    # Der Server kann intern auf eine andere Samplerate resamplen (modellabhängig)
    # -> Original-Mono UND die Regionen-Grenzen auf dieselbe Samplerate bringen,
    # damit die Rekonstruktion exakt zusammenpasst.
    if result_sr != orig_sr:
        mono_at_result_sr = librosa.resample(mono, orig_sr=orig_sr, target_sr=result_sr)
        scale = result_sr / orig_sr
        regions_scaled = [(int(round(s * scale)), int(round(e * scale))) for s, e in regions]
        total_at_result_sr = len(mono_at_result_sr)
    else:
        mono_at_result_sr = mono
        regions_scaled = regions
        total_at_result_sr = total_samples

    vocals_audio = reassemble_from_active(
        total_at_result_sr, regions_scaled, vocals_active, mono_at_result_sr, result_sr,
    )
    other_audio = reassemble_from_active(
        total_at_result_sr, regions_scaled, other_active, np.zeros_like(mono_at_result_sr), result_sr,
    )

    return vocals_audio, other_audio, result_sr, skip_fraction


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
        vocals_audio, other_audio, sr, skip_fraction = _separate_all(raw_path)
        pre_info = f"{skip_fraction * 100:.0f}% als Stille übersprungen" if skip_fraction > 0.01 else ""
        return finalize_vocals(
            raw_path, vocals_audio, other_audio, sr,
            target_sr=target_sr, denoise=denoise, music_mode=music_mode, pre_info=pre_info,
        )
    except requests.exceptions.ConnectionError:
        return DemucsResult(
            str(raw_path), "", False,
            "separator_server nicht erreichbar. Läuft er? (siehe start.sh / "
            "separator_server/install.sh, Port 8052)",
        )
    except Exception as e:
        return DemucsResult(str(raw_path), "", False, truncate_output(str(e)))


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
    return results
