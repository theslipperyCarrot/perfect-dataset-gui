"""
VAD-Segmentierung + Transkription in einem Schritt.

Ablauf pro Datei (aus data/processed):
  1. Silero VAD liefert grobe Sprachintervalle (nur als schneller Vorab-Check,
     ob überhaupt Sprache in der Datei ist).
  2. whisperX transkribiert die gesamte Datei mit Wort-Timestamps.
  3. Clip-Grenzen werden DIREKT aus den Wort-Timestamps gebaut (nicht mehr
     aus groben VAD-Intervallen + nachträglichem Snapping) — eine Grenze ist
     dadurch immer exakt eine Wortgrenze, kann also nie mitten im Wort
     landen. Jede ausreichend große Pause zwischen zwei Wörtern wird als
     Trennstelle genutzt, sobald der Clip schon lang genug ist.
  4. Jeder finale Clip wird als eigene WAV-Datei gespeichert, zusammen
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
    SILENCE_TRIM_TOP_DB, SILENCE_TRIM_MAX_S, FADE_DURATION_S,
)
from modules.torchaudio_compat import ensure_patched as ensure_torchaudio_compat

VAD_SAMPLE_RATE = 16000

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
# Schritt 2: Clip-Grenzen DIREKT aus den Wort-Timestamps bauen
# ---------------------------------------------------------------------
# Bewusst NICHT mehr über grobe VAD-Intervalle + nachträgliches Wortgrenzen-
# Snapping (das brauchte bei zu langen Abschnitten einen blinden Zeit-Hart-
# Split, der mitten im Wort schneiden konnte, siehe Git-Historie). Stattdessen
# läuft der Algorithmus direkt über die Wortliste: eine Clip-Grenze ist
# IMMER eine Wort-Grenze, kann also nie mitten im Wort landen. Getrennt wird,
# sobald der Clip schon lang genug ist (min_duration erreicht) UND entweder
# (a) das letzte Wort mit Satzende oder Komma endet, oder (b) eine
# ausreichend große Pause zum nächsten Wort folgt — (a) greift auch dann,
# wenn an dieser Stelle gar keine messbare Pause vorhanden ist (z.B. beim
# schnellen Vorlesen, wo an Kommas kaum pausiert wird).

def _ends_at_natural_break(word_text: str) -> bool:
    """True, wenn ein Wort mit Satzende- oder Komma-Interpunktion endet.
    whisperX hängt Satzzeichen direkt ans vorangehende Wort an (z.B.
    "angekündigt." oder "haben,"), daher reicht ein einfacher endswith-Check."""
    return word_text.strip().endswith((".", "!", "?", "…", ",", ";", ":"))


def _build_clips_from_words(
    words: list[dict],
    max_duration: float = MAX_CLIP_DURATION_S,
    min_duration: float = MIN_CLIP_DURATION_S,
    silence_gap: float = MIN_SILENCE_GAP_S,
) -> list[tuple[float, float]]:
    if not words:
        return []

    clips = []
    current: list[dict] = [words[0]]

    for word in words[1:]:
        clip_start = current[0]["start"]
        current_end = current[-1]["end"]
        current_duration = current_end - clip_start
        gap_before = word["start"] - current_end
        prospective_duration = word["end"] - clip_start

        close_here = False
        if prospective_duration > max_duration:
            # Dieses Wort noch mit reinzunehmen würde den Clip zu lang machen
            # -> auf jeden Fall vorher schließen (Grenze = Ende des letzten
            # Wortes, das noch reinpasst -> nie mitten im Wort).
            close_here = True
        elif current_duration >= min_duration and _ends_at_natural_break(current[-1]["text"]):
            # Satzende oder Komma erreicht, Clip schon lang genug -> hier
            # trennen, AUCH wenn keine messbare Pause folgt (beim schnellen
            # Vorlesen wird an Kommas oft kaum oder gar nicht pausiert, das
            # Satzzeichen selbst ist trotzdem eine sinnvolle Trennstelle).
            close_here = True
        elif gap_before >= silence_gap and current_duration >= min_duration:
            # Natürliche Pause gefunden UND der Clip ist schon lang genug ->
            # hier trennen, statt bis zum Maximum weiterzusammeln.
            close_here = True

        if close_here:
            clips.append((current[0]["start"], current[-1]["end"]))
            current = [word]
        else:
            current.append(word)

    if current:
        clips.append((current[0]["start"], current[-1]["end"]))

    # Clips unter der Mindestlänge mit dem vorherigen zusammenlegen, statt sie
    # zu verwerfen (kann bei erzwungenen Schnitten am Dateiende entstehen) —
    # aber NIEMALS über die Maximallänge hinaus, sonst wäre die 10s-Grenze
    # wieder ausgehebelt. Passt der kurze Rest nicht mehr rein, wird er weiter
    # unten einfach verworfen (besser ein fehlendes Wort als ein zu langer Clip).
    merged: list[tuple[float, float]] = []
    for start, end in clips:
        if merged and (end - start) < min_duration:
            prev_start, prev_end = merged[-1]
            if (end - prev_start) <= max_duration:
                merged[-1] = (prev_start, end)
                continue
        merged.append((start, end))

    return [(s, e) for s, e in merged if e - s >= min_duration]


# ---------------------------------------------------------------------
# Schritt 3: whisperX-Transkription
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
    darüber, um Klick-/Atem-Reste an Clip-Grenzen zu vermeiden.

    librosa.effects.trim() bewertet Lautstärke relativ zum LAUTESTEN Punkt im
    gesamten Clip. Ein leises, kurzes Wort direkt am Rand (z.B. ein
    unbetontes "Sie") kann dadurch fälschlich als Stille erkannt und komplett
    weggeschnitten werden, wenn es deutlich leiser ist als der Rest des Clips.
    Da die Clip-Grenzen ohnehin bereits exakt auf Wort-Timestamps sitzen
    (siehe _build_clips_from_words), gibt es kaum echte Stille zum
    Wegschneiden — daher wird pro Rand höchstens SILENCE_TRIM_MAX_S
    weggeschnitten, egal was librosa vorschlägt. Das reicht für Atem-/
    Klick-Reste, kann aber nie ein ganzes Wort verschlucken.
    """
    if len(clip_audio) == 0:
        return clip_audio

    _, index = librosa.effects.trim(clip_audio, top_db=SILENCE_TRIM_TOP_DB)
    max_trim = int(SILENCE_TRIM_MAX_S * sr)
    start_idx = min(int(index[0]), max_trim)
    end_idx = max(int(index[1]), len(clip_audio) - max_trim)
    trimmed = clip_audio[start_idx:end_idx]
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
    if not intervals:
        print(f"[segment] {processed_path.name}: keine Sprache erkannt -> 0 Clips")
        return []

    words = _transcribe_with_words(processed_path, device=device, language=language)
    if not words:
        print(f"[segment] {processed_path.name}: whisperX lieferte keine Wörter -> 0 Clips")
        return []

    clip_bounds = _build_clips_from_words(words)
    print(f"[segment] {processed_path.name}: {len(clip_bounds)} Clip(s) aus Wort-Timestamps "
          f"gebaut (max. {MAX_CLIP_DURATION_S}s, min. {MIN_CLIP_DURATION_S}s)")
    if not clip_bounds:
        return []

    # Audio laden (voller Arbeitsstandard, nicht die 16kHz-VAD-Kopie)
    audio, sr = sf.read(str(processed_path))
    if sr != WORKING_SAMPLE_RATE:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=WORKING_SAMPLE_RATE)
        sr = WORKING_SAMPLE_RATE

    entries = []
    for idx, (start, end) in enumerate(clip_bounds):
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


def process_all(
    device: str = "cuda", language: str = WHISPERX_LANGUAGE, progress_cb=None,
) -> tuple[list[ClipEntry], list[str]]:
    """progress_cb(index_1_based, total, filename) wird nach jeder Quelldatei aufgerufen.

    Das Manifest wird nach JEDER Quelldatei neu geschrieben (nicht erst am Ende
    des kompletten Batches), damit bei einem Abbruch mittendrin (GPU-OOM,
    Absturz, Server-Kill) nicht die Transkription aller bereits fertig
    verarbeiteten Dateien verloren geht — nur die noch offenen Dateien fehlen
    dann im Manifest, nicht der ganze Lauf. Schlägt eine einzelne Datei fehl,
    bricht das außerdem nicht den kompletten Batch ab: der Fehler wird geloggt,
    das Manifest bleibt auf dem letzten guten Stand, und die übrigen Dateien
    werden trotzdem weiterverarbeitet.

    Rückgabe: (entries, skipped_filenames) — skipped_filenames listet Dateien,
    die wegen eines Fehlers übersprungen wurden (leer im Normalfall).
    """
    processed_files = sorted(PROCESSED_DIR.glob("*.wav"))
    all_entries: list[ClipEntry] = []
    skipped: list[str] = []
    total = len(processed_files)
    for i, f in enumerate(processed_files, start=1):
        try:
            all_entries.extend(process_file(f, device=device, language=language))
        except Exception as e:
            print(f"[segment] {f.name}: Fehler bei der Verarbeitung, Datei wird übersprungen: {e}")
            skipped.append(f.name)
        else:
            save_manifest(all_entries)
        if progress_cb:
            progress_cb(i, total, f.name)

    return all_entries, skipped


def save_manifest(entries: list[ClipEntry]):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump([asdict(e) for e in entries], fh, ensure_ascii=False, indent=2)


def load_manifest() -> list[ClipEntry]:
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [ClipEntry(**r) for r in raw]
