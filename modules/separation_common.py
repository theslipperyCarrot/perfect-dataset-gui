"""
Gemeinsame Logik für alle Vocal-Separation-Engines (Demucs, UVR/audio-separator).

Beide Engines liefern am Ende dasselbe Zwischenergebnis: eine Vocals-Spur und
eine "Rest"-Spur (Musik/Instrumente/Rauschen). Alles, was NACH dieser
Trennung passiert — Stille-Sicherheitscheck, optionales Herausschneiden von
musik-dominierten Abschnitten, optionales DeepFilterNet-Denoising, finales
Speichern — ist Engine-unabhängig und lebt deshalb hier statt doppelt in
demucs_separation.py und uvr_separation.py.
"""
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import librosa

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, WORKING_SAMPLE_RATE
from modules import settings

SILENCE_RMS_THRESHOLD = 1e-4  # ~ -80 dBFS; deutlich unter jeder normalen Sprachaufnahme


@dataclass
class DemucsResult:
    """Name aus Kompatibilitätsgründen (v0.7-0.17 hieß die Engine nur
    "Demucs") beibehalten — wird von beiden Engines genutzt."""
    source_path: str
    target_path: str
    success: bool
    message: str = ""
    info: str = ""  # zusätzliche Erfolgs-Info (z.B. "12% wegen Musik herausgeschnitten")


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0


def _cut_music_sections(vocals: np.ndarray, music: np.ndarray, sr: int) -> tuple:
    """Schneidet Zeitfenster heraus, in denen die Musik-Restenergie einen zu
    hohen Anteil an der Gesamtenergie hat, statt die (evtl. unsaubere)
    Trennung dort zu behalten. Gibt (gekürztes_audio, entfernter_anteil) zurück."""
    window = max(1, int(settings.get("MUSIC_CUT_WINDOW_S") * sr))
    energy_ratio = settings.get("MUSIC_CUT_ENERGY_RATIO")
    n = len(vocals)
    kept_chunks = []
    cut_samples = 0
    for start in range(0, n, window):
        end = min(n, start + window)
        v_rms = _rms(vocals[start:end])
        m_rms = _rms(music[start:end])
        ratio = m_rms / (v_rms + m_rms + 1e-9)
        if ratio > energy_ratio:
            cut_samples += (end - start)
            continue
        kept_chunks.append(vocals[start:end])
    result = np.concatenate(kept_chunks) if kept_chunks else np.array([], dtype=np.float32)
    cut_fraction = (cut_samples / n) if n else 0.0
    return result, cut_fraction


def find_active_regions(
    mono_audio: np.ndarray, sr: int,
    window_s: float = None,
    min_silence_s: float = None,
    rms_threshold: float = None,
) -> list:
    """Liefert eine Liste von (start_sample, end_sample)-Regionen, die NICHT
    aus längerer, echter Stille bestehen — nur diese müssen durch die teure
    Trennung (Demucs/UVR). Bewusst KEIN Sprach-/Musik-Unterschied (das würde
    VAD brauchen, und VAD ist ausgerechnet bei lauter Hintergrundmusik
    unzuverlässig — genau dort, wo die Trennung am nötigsten ist). Stattdessen
    ein simpler Pegel-Check: alles, was klar über dem Rauschboden liegt
    (Sprache UND Musik gleichermaßen), zählt als "aktiv".

    window_s/min_silence_s/rms_threshold: None -> aktuelle Einstellung aus
    dem Einstellungen-Tab (settings.py) wird zur Laufzeit geholt. Explizite
    Werte (z.B. für Tests) übersteuern das.

    Kurze Pausen (< min_silence_s) werden bewusst NICHT herausgetrennt, um
    normale Sprechpausen zwischen Sätzen nicht aus dem Zusammenhang zu
    reißen — nur längere Stille-Strecken (Intro, Outro, lange Pausen)
    werden übersprungen.

    Bei einer Datei, die laut Schwelle KOMPLETT still wäre, wird sicherheits-
    halber die gesamte Datei als eine aktive Region zurückgegeben — lieber
    einmal unnötig trennen, als eine falsch als "still" erkannte Datei
    komplett zu überspringen."""
    if window_s is None:
        window_s = settings.get("SILENCE_SKIP_WINDOW_S")
    if min_silence_s is None:
        min_silence_s = settings.get("SILENCE_SKIP_MIN_DURATION_S")
    if rms_threshold is None:
        rms_threshold = settings.get("SILENCE_SKIP_RMS_THRESHOLD")

    window = max(1, int(window_s * sr))
    n = len(mono_audio)
    if n == 0:
        return []

    is_silent = [
        _rms(mono_audio[start:min(n, start + window)]) < rms_threshold
        for start in range(0, n, window)
    ]

    min_windows = max(1, int(round(min_silence_s / window_s)))
    active_mask = [True] * len(is_silent)
    i = 0
    while i < len(is_silent):
        if is_silent[i]:
            j = i
            while j < len(is_silent) and is_silent[j]:
                j += 1
            if (j - i) >= min_windows:
                for k in range(i, j):
                    active_mask[k] = False
            i = j
        else:
            i += 1

    regions = []
    i = 0
    while i < len(active_mask):
        if active_mask[i]:
            j = i
            while j < len(active_mask) and active_mask[j]:
                j += 1
            regions.append((i * window, min(n, j * window)))
            i = j
        else:
            i += 1

    return regions if regions else [(0, n)]


def build_active_only_buffer(audio: np.ndarray, regions: list) -> np.ndarray:
    """Baut einen zusammenhängenden Puffer NUR aus den aktiven Regionen, für
    einen einzigen, zusammenhängenden Trennungs-Durchlauf statt vieler
    kleiner Aufrufe. `audio` kann mono (1D) oder mehrkanalig (Kanäle zuerst,
    z.B. Demucs-Tensor-Layout) sein — Slicing erfolgt auf der letzten Achse."""
    if not regions:
        return audio[..., :0]
    return np.concatenate([audio[..., s:e] for s, e in regions], axis=-1)


def reassemble_from_active(
    full_length: int, regions: list, active_result: np.ndarray, silent_source: np.ndarray,
    sr: int, crossfade_s: float = None,
) -> np.ndarray:
    """Baut die Original-Zeitachse aus dem getrennten Ergebnis (nur für die
    aktiven Regionen vorhanden) und `silent_source` (unverändertes Original
    für die übersprungenen Stille-Abschnitte) wieder zusammen. Mit kurzem
    Crossfade an jedem Übergang, damit an den Nahtstellen kein hörbarer
    Klick entsteht. Längenabweichungen (z.B. durch Rundung beim Resampling
    zwischen Client- und Server-Samplerate) werden defensiv mit Stille
    aufgefüllt/abgeschnitten statt einen Fehler zu werfen.

    crossfade_s: None -> aktuelle Einstellung aus dem Einstellungen-Tab."""
    if crossfade_s is None:
        crossfade_s = settings.get("SILENCE_SKIP_CROSSFADE_S")
    out = np.zeros(full_length, dtype=np.float32)
    active_result = np.asarray(active_result, dtype=np.float32)
    silent_source = np.asarray(silent_source, dtype=np.float32)
    fade_len = max(1, int(crossfade_s * sr))

    def _safe_slice(src: np.ndarray, start: int, end: int) -> np.ndarray:
        start = max(0, min(start, len(src)))
        end = max(start, min(end, len(src)))
        chunk = src[start:end]
        want = end - start
        if len(chunk) < want:
            chunk = np.pad(chunk, (0, want - len(chunk)))
        return chunk

    pos = 0  # Position im active_result-Puffer
    prev_end = 0
    for start, end in regions:
        start = min(start, full_length)
        end = min(end, full_length)
        if start > prev_end:
            out[prev_end:start] = _safe_slice(silent_source, prev_end, start)

        length = end - start
        chunk = active_result[pos:pos + length]
        if len(chunk) < length:
            chunk = np.pad(chunk, (0, length - len(chunk)))
        out[start:end] = chunk
        pos += length

        # Crossfade am Übergang Stille -> aktiv (nur wenn davor tatsächlich
        # eine übersprungene Stille-Region lag, nicht am allerersten Sample)
        if start > 0 and start > prev_end - 1:
            f = min(fade_len, start, length)
            if f > 0:
                fade_in = np.linspace(0.0, 1.0, f)
                out[start:start + f] = (
                    out[start:start + f] * fade_in + _safe_slice(silent_source, start, start + f) * (1 - fade_in)
                )
        prev_end = end

    if prev_end < full_length:
        out[prev_end:full_length] = _safe_slice(silent_source, prev_end, full_length)
        # Crossfade am Übergang aktiv -> Stille
        f = min(fade_len, full_length - prev_end, prev_end)
        if f > 0:
            fade_out = np.linspace(1.0, 0.0, f)
            out[prev_end - f:prev_end] = (
                out[prev_end - f:prev_end] * fade_out
                + _safe_slice(silent_source, prev_end - f, prev_end) * (1 - fade_out)
            )

    return out



def finalize_vocals(
    raw_path: Path, vocals_audio: np.ndarray, other_audio: np.ndarray, sr: int,
    target_sr: int = WORKING_SAMPLE_RATE, denoise: bool = True,
    music_mode: str = "remove_music", pre_info: str = "",
) -> DemucsResult:
    """Gemeinsamer Rest der Pipeline nach der reinen Trennung: Musik-Abschnitte
    optional herausschneiden, Stille-Sicherheitschecks, optionales Denoising,
    Resampling + finales Speichern nach PROCESSED_DIR. `other_audio` ist die
    Summe aller Nicht-Vocals-Stems (bei Demucs: drums+bass+other, bei UVR:
    die eine "Instrumental"-Spur). `pre_info` wird der Info-Meldung
    vorangestellt (z.B. wie viel Prozent der Datei als Stille übersprungen
    wurde, siehe find_active_regions())."""
    info = pre_info

    if music_mode == "cut_music_sections":
        vocals_audio, cut_fraction = _cut_music_sections(vocals_audio, other_audio, sr)
        if info:
            info += "; "
        info += f"{cut_fraction * 100:.0f}% wegen Musik herausgeschnitten"
        if len(vocals_audio) == 0:
            return DemucsResult(
                str(raw_path), "", False,
                "Nach dem Herausschneiden der Musik-Abschnitte blieb nichts übrig — "
                "vermutlich ist die ganze Datei musikdominiert.",
            )

    rms_after_separation = _rms(vocals_audio)
    if rms_after_separation < SILENCE_RMS_THRESHOLD:
        return DemucsResult(
            str(raw_path), "", False,
            f"Ergebnis nach der Trennung ist verdächtig leise/leer (RMS={rms_after_separation:.2e}) — "
            "vermutlich ein Fehler in der Verarbeitung statt eines echten Ergebnisses, "
            "kein 'Erfolg' zum Sicherheitscheck. Bitte Ausgangsdatei/Terminal-Log prüfen.",
        )

    if denoise:
        from modules.denoise import denoise_file
        DEMUCS_TMP_DIR = PROCESSED_DIR / "_demucs_tmp"
        DEMUCS_TMP_DIR.mkdir(parents=True, exist_ok=True)
        tmp_in = DEMUCS_TMP_DIR / f"{raw_path.stem}_vocals.wav"
        sf.write(str(tmp_in), vocals_audio, sr, subtype="PCM_16")

        denoised_path = DEMUCS_TMP_DIR / f"{raw_path.stem}_denoised.wav"
        denoise_file(tmp_in, denoised_path)
        vocals_audio, sr = sf.read(str(denoised_path))

        rms_after_denoise = _rms(np.asarray(vocals_audio))
        if rms_after_denoise < SILENCE_RMS_THRESHOLD:
            return DemucsResult(
                str(raw_path), "", False,
                f"Ergebnis nach DeepFilterNet-Denoising ist verdächtig leise/leer "
                f"(RMS={rms_after_denoise:.2e}) — die Trennung war noch ok, das "
                "Denoising hat das Signal aber offenbar komplett weggefiltert.",
            )

    if sr != target_sr:
        vocals_audio = librosa.resample(
            np.asarray(vocals_audio, dtype=np.float32), orig_sr=sr, target_sr=target_sr,
        )

    target_path = PROCESSED_DIR / f"{raw_path.stem}.wav"
    sf.write(str(target_path), vocals_audio, target_sr, subtype="PCM_16")

    return DemucsResult(str(raw_path), str(target_path), True, info=info)


def truncate_output(text: str, head: int = 800, tail: int = 1200) -> str:
    """Behält Anfang UND Ende einer langen Fehlerausgabe, statt nur das Ende —
    wichtige frühe Meldungen (z.B. fehlende Abhängigkeiten) gehen sonst verloren."""
    if len(text) <= head + tail:
        return text
    return f"{text[:head]}\n[... gekürzt ...]\n{text[-tail:]}"


def list_processed_files() -> list:
    return sorted(PROCESSED_DIR.glob("*.wav"))
