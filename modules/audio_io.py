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
    message: str = ""  # bei Fehlschlag: Grund, wird in der GUI angezeigt


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Mehrkanal-Audio auf Mono reduzieren (Mittelwert über Kanäle)."""
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def _load_with_pydub(source: Path):
    """Fallback über FFmpeg (via pydub), falls soundfile/librosa scheitert.
    Deckt so gut wie jede Kodierung ab — auch z.B. eine falsch als .wav
    benannte MP3-Datei oder exotische WAV-Subtypen, die libsndfile nicht
    kennt (manche ältere Aufnahmegeräte/-programme nutzen sowas)."""
    from pydub import AudioSegment
    seg = AudioSegment.from_file(str(source))
    sr = seg.frame_rate
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    if seg.channels > 1:
        samples = samples.reshape((-1, seg.channels)).mean(axis=1)
    max_val = float(1 << (8 * seg.sample_width - 1))
    audio_mono = samples / max_val
    return audio_mono, sr


def _load_raw_pcm_fallback(source: Path):
    """Letzter Rettungsanker: WAV-Header manuell auslesen (Format/Samplerate/
    Kanäle/Bit-Tiefe) und die TATSÄCHLICHE Restlänge der Datei als PCM-Daten
    nehmen, statt der im Header angegebenen Datenchunk-Größe zu vertrauen.
    Nötig, weil weder soundfile noch ffmpeg von sich aus über eine falsche/
    0 angegebene Datenchunk-Größe hinweglesen (typisch bei nicht sauber
    abgeschlossenen/finalisierten Aufnahmen — z.B. wenn die Aufnahme-Software
    abgestürzt ist, bevor sie die Header-Größe nachträglich korrigieren konnte)."""
    import struct
    raw = source.read_bytes()

    if raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("Keine gültige RIFF/WAVE-Datei")

    pos = 12
    fmt = None
    data_start = None
    while pos + 8 <= len(raw):
        chunk_id = raw[pos:pos + 4]
        chunk_size = struct.unpack_from("<I", raw, pos + 4)[0]
        chunk_data_start = pos + 8
        if chunk_id == b"fmt ":
            _fmt_tag, channels, sample_rate, _byte_rate, _block_align, bits_per_sample = \
                struct.unpack_from("<HHIIHH", raw, chunk_data_start)
            fmt = (channels, sample_rate, bits_per_sample)
        elif chunk_id == b"data":
            data_start = chunk_data_start
            break
        pos = chunk_data_start + chunk_size + (chunk_size % 2)  # Chunks sind wortalignt

    if fmt is None or data_start is None:
        raise ValueError("fmt- oder data-Chunk im Header nicht gefunden")

    channels, sample_rate, bits_per_sample = fmt
    bytes_per_sample = bits_per_sample // 8
    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    if bytes_per_sample not in dtype_map or channels < 1:
        raise ValueError(f"Nicht unterstütztes PCM-Format: {bits_per_sample} Bit, {channels} Kanäle")

    # TATSÄCHLICHE Restlänge der Datei nutzen, nicht die (evtl. falsche) Chunk-Größe
    actual_data_bytes = len(raw) - data_start
    n_frames = actual_data_bytes // (bytes_per_sample * channels)
    usable_bytes = n_frames * bytes_per_sample * channels
    if usable_bytes <= 0:
        raise ValueError("Auch nach dem data-Chunk-Header sind keine Nutzdaten vorhanden")

    samples = np.frombuffer(raw[data_start:data_start + usable_bytes], dtype=dtype_map[bytes_per_sample])
    samples = samples.astype(np.float32) / float(1 << (bits_per_sample - 1))
    if channels > 1:
        samples = samples.reshape((-1, channels)).mean(axis=1)

    return samples, sample_rate


def _load_audio_any(source: Path):
    """Lädt Audio robust in drei Stufen:
    1. soundfile/librosa (schnell, Standardfall)
    2. FFmpeg-Fallback über pydub (deckt exotische Kodierungen/falsch
       benannte Dateien ab)
    3. Manuelles Header-Parsing mit tatsächlicher Datei-Restlänge statt der
       angegebenen Chunk-Größe (deckt nicht finalisierte Aufnahmen mit
       kaputter/0 Datenchunk-Größe ab, wo weder 1. noch 2. helfen)
    Ein Ergebnis wird nur akzeptiert, wenn es auch zur Dateigröße passt —
    sonst wird die nächste Stufe versucht, statt eine unplausibel kurze
    Datei stillschweigend zu übernehmen."""
    file_size = source.stat().st_size
    attempts = []

    def _is_plausible(audio_mono, sr):
        if len(audio_mono) == 0:
            return False
        duration = len(audio_mono) / sr if sr else 0.0
        if file_size < 20_000:
            return True  # zu klein, um eine sinnvolle Bytes/s-Aussage zu treffen
        implied_bytes_per_s = file_size / duration if duration > 0 else float("inf")
        # Keine reale Audioformat-Kombination (auch keine sehr hochauflösende)
        # braucht mehr als ~1 MB/s -> deutlich darüber = Header hat gelogen.
        return implied_bytes_per_s < 1_000_000

    for name, loader in [
        ("soundfile/librosa", lambda: _to_mono_pair(*librosa.load(str(source), sr=None, mono=False))),
        ("FFmpeg (pydub)", lambda: _load_with_pydub(source)),
        ("manuelles Header-Parsing", lambda: _load_raw_pcm_fallback(source)),
    ]:
        try:
            audio_mono, sr = loader()
            if _is_plausible(audio_mono, sr):
                return audio_mono, sr
            duration = len(audio_mono) / sr if sr else 0.0
            attempts.append(f"{name}: nur {duration:.2f}s bei {file_size / 1024:.0f} KB Dateigröße (unplausibel)")
        except Exception as e:
            attempts.append(f"{name}: {e}")
        if name != "manuelles Header-Parsing":
            print(f"[audio_io] {name} fehlgeschlagen/unplausibel für {source.name}, versuche nächste Stufe...")

    raise RuntimeError(f"Konnte Audio nicht laden ({source.name}). Versuche: " + " | ".join(attempts))


def _to_mono_pair(audio, sr):
    return (_to_mono(audio) if audio.ndim > 1 else audio), sr


def import_audio_file(source_path: str, target_sr: int = WORKING_SAMPLE_RATE) -> ImportResult:
    """
    Lädt eine einzelne Audiodatei, konvertiert auf Mono + Ziel-Samplerate
    und speichert sie als WAV im RAW-Verzeichnis.

    Bewusst konservativ: keine Lautstärke-Normalisierung an dieser Stelle,
    das passiert erst nach der Demucs-Trennung, damit Normalisierung nicht
    durch Musik-/Rauschanteile verzerrt wird.
    """
    source = Path(source_path)
    audio, orig_sr = _load_audio_any(source)

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
                original_sr=0, target_sr=target_sr, message=str(e),
            ))
            print(f"[audio_io] Fehler beim Import von {p}: {e}")
    return results


def list_raw_files() -> list[Path]:
    return sorted(RAW_DIR.glob("*.wav"))
