"""
VAD-Segmentierung + Transkription in einem Schritt.

Ablauf pro Datei (aus data/processed):
  1. Silero VAD liefert grobe Sprachintervalle (schnell, robust gegen
     Atmer/Stille, aber nicht wortgenau).
  2. Die groben Intervalle werden zu Clips gruppiert, die Mindest-/
     Maximallänge und Mindest-Stille-Puffer aus config.py einhalten.
  3. whisperX transkribiert die gesamte Datei mit Wort-Timestamps.
  4. Die Clip-Grenzen aus Schritt 2 werden auf die nächstgelegene
     Wortgrenze aus Schritt 3 "eingerastet", damit kein Wort mitten
     durchgeschnitten wird.
  5. Jeder finale Clip wird als eigene WAV-Datei gespeichert, zusammen
     mit einem manifest.json, das Text + Zeiten + Review-Status je
     Clip enthält (Grundlage für den Review-Tab).

Silero VAD arbeitet nur mit 8/16 kHz -> für die VAD-Analyse wird intern
auf 16 kHz resampled, die eigentlichen Clips werden aber weiterhin aus
dem 22050-Hz-Arbeitsmaterial geschnitten, um keine Qualität zu verlieren.
"""
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
import soundfile as sf
import librosa
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    PROCESSED_DIR, SEGMENTS_DIR, WORKING_SAMPLE_RATE,
    MIN_CLIP_DURATION_S, MAX_CLIP_DURATION_S, MIN_SILENCE_GAP_S,
    SILERO_VAD_THRESHOLD, WHISPERX_MODEL, WHISPERX_LANGUAGE, WHISPERX_COMPUTE_TYPE,
    SILENCE_TRIM_TOP_DB, FADE_DURATION_S,
)
from modules.torchaudio_compat import ensure_patched as ensure_torchaudio_compat

VAD_SAMPLE_RATE = 16000
WORD_SNAP_TOLERANCE_S = 0.35  # max. Abstand, um eine Clip-Grenze auf ein Wortende zu snappen

MANIFEST_PATH = SEGMENTS_DIR / "manifest.json"

_silero_model = None
_silero_utils = None
_whisperx_model = None
_whisperx_align_model = None
_whisperx_align_metadata = None
_whisperx_language_loaded = None  # welche Sprache aktuell geladen ist (für Cache-Invalidierung)


@dataclass
class ClipEntry:
    clip_filename: str
    source_file: str
    start_s: float
    end_s: float
    duration_s: float
    text: str
    approved: bool = None  # None = noch nicht reviewed, True/False nach Review-Tab
    id: str = field(default="")
    asr_confidence: float = None  # Ø whisperX-Wort-Konfidenz im Clip (0-1)
    nisqa_mos: float = None       # NISQA Mean-Opinion-Score (1-5), None = noch nicht bewertet


# ---------------------------------------------------------------------
# Modelle laden (lazy, damit die GUI startet ohne dass sofort GPU-Modelle
# geladen werden — erst wenn tatsächlich verarbeitet wird)
# ---------------------------------------------------------------------

def _load_silero():
    global _silero_model, _silero_utils
    if _silero_model is None:
        ensure_torchaudio_compat()
        _silero_model, _silero_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
    return _silero_model, _silero_utils


def _load_whisperx(device: str = "cuda", language: str = WHISPERX_LANGUAGE):
    global _whisperx_model, _whisperx_align_model, _whisperx_align_metadata, _whisperx_language_loaded
    ensure_torchaudio_compat()
    import whisperx

    if _whisperx_language_loaded != language:
        # Sprache gewechselt -> Cache verwerfen, sonst würde mit dem
        # Modell/Alignment der vorherigen Sprache weitertranskribiert.
        _whisperx_model = None
        _whisperx_align_model = None
        _whisperx_align_metadata = None
        _whisperx_language_loaded = language

    if _whisperx_model is None:
        _whisperx_model = whisperx.load_model(
            WHISPERX_MODEL, device=device, compute_type=WHISPERX_COMPUTE_TYPE,
            language=language,
        )
    if _whisperx_align_model is None:
        _whisperx_align_model, _whisperx_align_metadata = whisperx.load_align_model(
            language_code=language, device=device,
        )
    return _whisperx_model, _whisperx_align_model, _whisperx_align_metadata


# ---------------------------------------------------------------------
# Schritt 1: grobe VAD-Sprachintervalle
# ---------------------------------------------------------------------

def _get_vad_speech_intervals(audio_path: Path) -> list[tuple[float, float]]:
    """Lädt selbst über soundfile/librosa (statt Sileros eigener read_audio()-
    Hilfsfunktion, die intern kaputtes torchaudio nutzt — gleiches Muster wie
    bei Demucs/DeepFilterNet, siehe demucs_separation.py/denoise.py)."""
    model, utils = _load_silero()
    get_speech_timestamps, *_ = utils

    audio, sr = sf.read(str(audio_path), always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)  # Mono
    if sr != VAD_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=VAD_SAMPLE_RATE)
    wav = torch.from_numpy(np.ascontiguousarray(audio))

    timestamps = get_speech_timestamps(
        wav, model,
        sampling_rate=VAD_SAMPLE_RATE,
        threshold=SILERO_VAD_THRESHOLD,
    )
    return [(t["start"] / VAD_SAMPLE_RATE, t["end"] / VAD_SAMPLE_RATE) for t in timestamps]


# ---------------------------------------------------------------------
# Schritt 2: grobe Intervalle zu Clips gruppieren (Min/Max-Länge, Silence-Gap)
# ---------------------------------------------------------------------

def _group_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []

    clips = []
    cur_start, cur_end = intervals[0]

    for start, end in intervals[1:]:
        gap = start - cur_end
        would_be_duration = end - cur_start

        if gap < MIN_SILENCE_GAP_S and would_be_duration <= MAX_CLIP_DURATION_S:
            # zu wenig Stille dazwischen und Clip würde nicht zu lang -> zusammenfassen
            cur_end = end
            continue

        # aktuellen Clip abschließen, falls lang genug
        if cur_end - cur_start >= MIN_CLIP_DURATION_S:
            clips.append((cur_start, cur_end))
        cur_start, cur_end = start, end

    if cur_end - cur_start >= MIN_CLIP_DURATION_S:
        clips.append((cur_start, cur_end))

    # zu lange Clips (kein passender Stille-Punkt gefunden) hart aufteilen
    final_clips = []
    for start, end in clips:
        duration = end - start
        if duration <= MAX_CLIP_DURATION_S:
            final_clips.append((start, end))
        else:
            n_parts = int(np.ceil(duration / MAX_CLIP_DURATION_S))
            part_len = duration / n_parts
            for i in range(n_parts):
                final_clips.append((start + i * part_len, start + (i + 1) * part_len))

    return final_clips


# ---------------------------------------------------------------------
# Schritt 3+4: whisperX-Transkription + Wortgrenzen-Snapping
# ---------------------------------------------------------------------

WHISPER_SAMPLE_RATE = 16000


def _load_audio_for_whisperx(path: Path) -> np.ndarray:
    """Lädt selbst über soundfile/librosa statt whisperx.load_audio() zu nutzen
    — gleiches Muster wie bei Demucs/DeepFilterNet/Silero VAD: die eigene
    Lade-Routine der jeweiligen Bibliothek nutzt oft veraltete torchaudio-APIs
    (hier vermutlich für Metadaten via torchaudio.AudioMetaData), die es in
    aktuellen torchaudio-Versionen nicht mehr gibt. whisperX braucht am Ende
    nur ein float32-Mono-Array bei 16 kHz, das laden wir zuverlässig selbst."""
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32)
    if sr != WHISPER_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=WHISPER_SAMPLE_RATE)
    return np.ascontiguousarray(audio)


def _transcribe_with_words(audio_path: Path, device: str = "cuda", language: str = WHISPERX_LANGUAGE) -> list[dict]:
    """Gibt eine flache Liste von Wörtern mit start/end/text zurück."""
    ensure_torchaudio_compat()
    import whisperx
    model, align_model, align_metadata = _load_whisperx(device=device, language=language)

    audio = _load_audio_for_whisperx(audio_path)
    result = model.transcribe(audio, language=language)
    aligned = whisperx.align(
        result["segments"], align_model, align_metadata, audio, device=device,
    )

    words = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({
                    "start": w["start"], "end": w["end"], "text": w["word"],
                    "score": w.get("score"),
                })
    return words


def _snap_to_word_boundary(t: float, words: list[dict], is_start: bool) -> float:
    """Verschiebt eine Zeitmarke auf die nächstgelegene Wortgrenze, falls
    innerhalb der Toleranz eine existiert — verhindert abgeschnittene Wörter."""
    best = t
    best_dist = WORD_SNAP_TOLERANCE_S
    for w in words:
        candidate = w["start"] if is_start else w["end"]
        dist = abs(candidate - t)
        if dist < best_dist:
            best_dist = dist
            best = candidate
    return best


def _text_for_range(words: list[dict], start: float, end: float) -> str:
    return " ".join(w["text"].strip() for w in words if w["start"] >= start - 0.05 and w["end"] <= end + 0.05).strip()


def _confidence_for_range(words: list[dict], start: float, end: float) -> float:
    """Ø whisperX-Wort-Konfidenz im Clip-Zeitraum (None, falls keine Scores vorliegen)."""
    scores = [
        w["score"] for w in words
        if w["start"] >= start - 0.05 and w["end"] <= end + 0.05 and w.get("score") is not None
    ]
    return round(sum(scores) / len(scores), 3) if scores else None


def _trim_and_fade(clip_audio: np.ndarray, sr: int) -> np.ndarray:
    """Schneidet Stille an den Rändern ab und legt einen kurzen Fade-in/-out
    darüber, um Klick-/Atem-Reste an Clip-Grenzen zu vermeiden."""
    if len(clip_audio) == 0:
        return clip_audio

    trimmed, _ = librosa.effects.trim(clip_audio, top_db=SILENCE_TRIM_TOP_DB)
    if len(trimmed) == 0:
        trimmed = clip_audio  # komplett unter der Schwelle -> lieber nichts wegschneiden

    fade_len = min(int(FADE_DURATION_S * sr), len(trimmed) // 2)
    if fade_len > 0:
        fade_in = np.linspace(0.0, 1.0, fade_len)
        fade_out = np.linspace(1.0, 0.0, fade_len)
        trimmed[:fade_len] *= fade_in
        trimmed[-fade_len:] *= fade_out

    return trimmed


# ---------------------------------------------------------------------
# Öffentliche Hauptfunktion
# ---------------------------------------------------------------------

def process_file(processed_path: Path, device: str = "cuda", language: str = WHISPERX_LANGUAGE) -> list[ClipEntry]:
    intervals = _get_vad_speech_intervals(processed_path)
    print(f"[segment] {processed_path.name}: {len(intervals)} VAD-Sprachintervall(e) gefunden")

    coarse_clips = _group_intervals(intervals)
    print(f"[segment] {processed_path.name}: {len(coarse_clips)} Clip(s) nach Gruppierung "
          f"(min. {MIN_CLIP_DURATION_S}s pro Clip)")
    if not coarse_clips:
        print(f"[segment] {processed_path.name}: keine verwertbaren Sprachabschnitte "
              f"(entweder keine Sprache erkannt, oder alle Abschnitte kürzer als "
              f"{MIN_CLIP_DURATION_S}s) -> 0 Clips")
        return []

    words = _transcribe_with_words(processed_path, device=device, language=language)

    # Grenzen an Wortgrenzen snappen
    snapped_clips = []
    for start, end in coarse_clips:
        snapped_start = _snap_to_word_boundary(start, words, is_start=True)
        snapped_end = _snap_to_word_boundary(end, words, is_start=False)
        if snapped_end - snapped_start >= MIN_CLIP_DURATION_S:
            snapped_clips.append((snapped_start, snapped_end))
    print(f"[segment] {processed_path.name}: {len(snapped_clips)} Clip(s) nach "
          f"Wortgrenzen-Snapping übrig")

    # Audio laden (voller Arbeitsstandard, nicht die 16kHz-VAD-Kopie)
    audio, sr = sf.read(str(processed_path))
    if sr != WORKING_SAMPLE_RATE:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=WORKING_SAMPLE_RATE)
        sr = WORKING_SAMPLE_RATE

    entries = []
    for idx, (start, end) in enumerate(snapped_clips):
        start_sample = max(0, int(start * sr))
        end_sample = min(len(audio), int(end * sr))
        clip_audio = audio[start_sample:end_sample]
        clip_audio = _trim_and_fade(np.asarray(clip_audio, dtype=np.float32), sr)

        clip_id = f"{processed_path.stem}_{idx:03d}"
        clip_filename = f"{clip_id}.wav"
        clip_path = SEGMENTS_DIR / clip_filename
        sf.write(str(clip_path), clip_audio, sr, subtype="PCM_16")

        text = _text_for_range(words, start, end)
        confidence = _confidence_for_range(words, start, end)

        entries.append(ClipEntry(
            id=clip_id,
            clip_filename=clip_filename,
            source_file=processed_path.name,
            start_s=round(start, 3),
            end_s=round(end, 3),
            asr_confidence=confidence,
            duration_s=round(len(clip_audio) / sr, 3),
            text=text,
        ))

    return entries


def process_all(device: str = "cuda", language: str = WHISPERX_LANGUAGE, progress_cb=None) -> list[ClipEntry]:
    """progress_cb(index_1_based, total, filename) wird nach jeder Quelldatei aufgerufen."""
    processed_files = sorted(PROCESSED_DIR.glob("*.wav"))
    all_entries: list[ClipEntry] = []
    total = len(processed_files)
    for i, f in enumerate(processed_files, start=1):
        all_entries.extend(process_file(f, device=device, language=language))
        if progress_cb:
            progress_cb(i, total, f.name)

    save_manifest(all_entries)
    return all_entries


def save_manifest(entries: list[ClipEntry]):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump([asdict(e) for e in entries], fh, ensure_ascii=False, indent=2)


def load_manifest() -> list[ClipEntry]:
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [ClipEntry(**r) for r in raw]
