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
import gc
import json
import sys
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, SEGMENTS_DIR, WORKING_SAMPLE_RATE, WHISPERX_MODEL, WHISPERX_LANGUAGE, WHISPERX_COMPUTE_TYPE
from modules import settings
from modules.text_compare import word_error_rate
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
    asr_confidence: float = None  # Round-Trip-Score (0-1, 1=perfekte Übereinstimmung) — siehe modules/roundtrip_check.py; None = noch nicht geprüft
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
        threshold=settings.get("SILERO_VAD_THRESHOLD"),
    )
    return [(t["start"] / VAD_SAMPLE_RATE, t["end"] / VAD_SAMPLE_RATE) for t in timestamps]


def _clamp_words_to_vad(words: list[dict], intervals: list[tuple[float, float]]) -> list[dict]:
    """Begrenzt jede Wort-Zeitmarke auf das VAD-Sprachintervall, in das ihr
    Wortanfang fällt.

    Grund: whisperX' eigene Wortausrichtung (wav2vec2-basiert) kann bei
    längeren Pausen/Atemgeräuschen erheblich danebenliegen — sie kann das
    letzte Wort vor einer Pause über die komplette Pause hinweg "verschmieren",
    bis weit in den nächsten Sprechabschnitt hinein (Praxisbeispiel: "wollen."
    endete laut whisperX weit hinter der tatsächlichen Pause, inklusive
    Atemgeräusch UND dem ersten Wort des nächsten Satzes). Silero VAD ist rein
    auf Sprache-/Stille-Erkennung spezialisiert (kein Alignment) und deutlich
    zuverlässiger, WO Sprache aufhört — wird hier als unabhängige zweite
    Meinung genutzt, um genau solche Ausreißer zu kappen, statt whisperX'
    Zeitstempeln blind zu vertrauen."""
    if not intervals:
        return words

    clamped = []
    for w in words:
        w = dict(w)  # Kopie, Original-Liste nicht verändern
        interval = None
        for s, e in intervals:
            if s <= w["start"] <= e:
                interval = (s, e)
                break
        if interval is None:
            # Wortanfang fällt in eine VAD-Stille-Lücke -> nächstgelegenes
            # Intervall nehmen statt das Wort ungeprüft zu übernehmen
            interval = min(
                intervals,
                key=lambda iv: min(abs(iv[0] - w["start"]), abs(iv[1] - w["start"])),
            )

        s, e = interval
        if w["end"] > e:
            w["end"] = e
        if w["start"] < s:
            w["start"] = s
        if w["end"] < w["start"]:  # entarteter Fall nach dem Kappen vermeiden
            w["end"] = w["start"]
        clamped.append(w)
    return clamped


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


def _pad_clip_boundaries(
    clips: list[tuple[float, float]], pad_s: float,
) -> list[tuple[float, float]]:
    """Erweitert jeden Clip um eine kleine Sicherheits-Vorlaufzeit vor dem
    ersten und nach dem letzten Wort. Grund: whisperX' Wortausrichtung ist am
    Rand nicht perfekt — der gemeldete Wort-Timestamp kann leicht zu spät
    (Wortanfang) bzw. zu früh (Wortende) liegen. Ohne Polster fehlt dann im
    Audio der Anfang eines Wortes, obwohl es im Transkript korrekt steht.

    Jede Seite bekommt das VOLLE Polster, unabhängig davon, wie groß die
    Lücke zum Nachbar-Clip ist. Das kann bei sehr eng benachbarten oder
    pausenlos aneinandergrenzenden Clips (siehe _ends_at_natural_break —
    seit 0.18.0 wird auch OHNE messbare Pause an Satzzeichen getrennt) zu
    einer minimalen Überlappung führen: ein paar hundertstel Sekunden
    Audiomaterial tauchen dann am Ende des einen UND am Anfang des nächsten
    Clips auf. Das ist bewusst in Kauf genommen — unhörbar/belanglos fürs
    Training, im Gegensatz zu einem abgeschnittenen oder fehlenden Wort.
    Frühere Version hat das Polster stattdessen auf die vorhandene Lücke
    begrenzt (nie Überlappung) — genau an den jetzt häufigen pausenlosen
    Satzzeichen-Schnitten gab es dadurch aber GAR kein Polster, weil die
    Lücke dort oft 0 ist."""
    if not clips or pad_s <= 0:
        return clips

    starts = [c[0] for c in clips]
    ends = [c[1] for c in clips]

    for i in range(len(clips) - 1):
        ends[i] += pad_s
        starts[i + 1] = max(0.0, starts[i + 1] - pad_s)

    starts[0] = max(0.0, starts[0] - pad_s)
    ends[-1] = ends[-1] + pad_s

    return list(zip(starts, ends))


def _build_clips_from_words(
    words: list[dict],
    max_duration: float,
    min_duration: float,
    silence_gap: float,
    pad_s: float,
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

    return _pad_clip_boundaries([(s, e) for s, e in merged if e - s >= min_duration], pad_s)


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


def _transcribe_with_words(
    audio_path: Path, device: str = "cuda", language: str = WHISPERX_LANGUAGE, batch_size: int = 8,
) -> list[dict]:
    """Gibt eine flache Liste von Wörtern mit start/end/text zurück.

    batch_size: whisperX verarbeitet mehrere Audio-Chunks gleichzeitig, um
    schneller zu sein — kostet aber proportional mehr VRAM. whisperX' eigener
    Default (32) ist für viele Consumer-GPUs zu hoch, gerade wenn (wie hier)
    parallel noch andere Modelle/Server auf derselben GPU laufen -> "CUDA
    failed with error out of memory" bzw. "batch_size probably too large".
    Einstellbar im Einstellungen-Tab (WHISPERX_BATCH_SIZE)."""
    ensure_torchaudio_compat()
    import whisperx
    model, align_model, align_metadata = _load_whisperx(device=device, language=language)

    audio = _load_audio_for_whisperx(audio_path)
    result = model.transcribe(audio, language=language, batch_size=batch_size)
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



def _trim_and_fade(clip_audio: np.ndarray, sr: int, top_db: float, max_trim_s: float, fade_s: float) -> np.ndarray:
    """Schneidet Stille an den Rändern ab und legt einen kurzen Fade-in/-out
    darüber, um Klick-/Atem-Reste an Clip-Grenzen zu vermeiden.

    librosa.effects.trim() bewertet Lautstärke relativ zum LAUTESTEN Punkt im
    gesamten Clip. Ein leises, kurzes Wort direkt am Rand (z.B. ein
    unbetontes "Sie") kann dadurch fälschlich als Stille erkannt und komplett
    weggeschnitten werden, wenn es deutlich leiser ist als der Rest des Clips.
    Da die Clip-Grenzen ohnehin bereits exakt auf Wort-Timestamps sitzen
    (siehe _build_clips_from_words) und zusätzlich ein Sicherheitspolster
    haben (WORD_BOUNDARY_PAD_S), gibt es kaum echte Stille zum Wegschneiden —
    daher wird pro Rand höchstens max_trim_s weggeschnitten, egal was librosa
    vorschlägt. Das reicht für Atem-/Klick-Reste, kann aber nie ein ganzes
    Wort verschlucken.
    """
    if len(clip_audio) == 0:
        return clip_audio

    _, index = librosa.effects.trim(clip_audio, top_db=top_db)
    max_trim = int(max_trim_s * sr)
    start_idx = min(int(index[0]), max_trim)
    end_idx = max(int(index[1]), len(clip_audio) - max_trim)
    trimmed = clip_audio[start_idx:end_idx]
    if len(trimmed) == 0:
        trimmed = clip_audio  # komplett unter der Schwelle -> lieber nichts wegschneiden

    fade_len = min(int(fade_s * sr), len(trimmed) // 2)
    if fade_len > 0:
        fade_in = np.linspace(0.0, 1.0, fade_len)
        fade_out = np.linspace(1.0, 0.0, fade_len)
        trimmed[:fade_len] *= fade_in
        trimmed[-fade_len:] *= fade_out

    return trimmed


def _find_best_matching_span(words: list[dict], expected_text: str):
    """Durchsucht `words` (Wort-Zeitstempel aus einem ERWEITERTEN Zeitfenster,
    siehe reclip_entry) nach der zusammenhängenden Wortfolge, die am besten
    zu `expected_text` passt (niedrigste Wort-Fehlerrate). Statt blind der
    ERSTEN Ausrichtung zu vertrauen (das war das ursprüngliche Problem),
    probiert diese Suche gezielt Fenstergrößen nahe der erwarteten Wortzahl
    durch — bei typischen Clip-Längen (wenige bis ein paar Dutzend Wörter im
    erweiterten Suchfenster) ist das schnell genug für eine einfache
    Brute-Force-Suche.

    Gibt (start_s, end_s, restfehler_wer) der besten Übereinstimmung zurück,
    oder None, falls `words` leer ist."""
    if not words:
        return None

    expected_word_count = len([w for w in expected_text.split() if w.strip()])
    if expected_word_count == 0:
        return None

    n = len(words)
    min_len = max(1, expected_word_count - 2)
    max_len = min(n, expected_word_count + 3)

    best = None  # (wer, i, j)
    for length in range(min_len, max_len + 1):
        for i in range(0, n - length + 1):
            j = i + length
            span_text = " ".join(w["text"] for w in words[i:j])
            wer = word_error_rate(expected_text, span_text)
            if best is None or wer < best[0]:
                best = (wer, i, j)
            if wer == 0.0:
                break  # exakter Treffer -> keine bessere Übereinstimmung mehr möglich
        if best is not None and best[0] == 0.0:
            break

    if best is None:
        return None
    wer, i, j = best
    return words[i]["start"], words[j - 1]["end"], wer


def reclip_entry(
    entry: "ClipEntry", window_start: float, window_end: float,
    device: str = "cuda", language: str = WHISPERX_LANGUAGE, pad_s: float = None,
):
    """Versucht, EINEN einzelnen Clip anhand eines erweiterten Zeitfensters
    aus der Quelldatei neu zu schneiden.

    Grund: Fehlt am Rand ein Wort, ist die fehlende Audio-Information im
    bereits geschnittenen Clip unwiederbringlich weg — dafür muss auf die
    Original-Quelldatei (data/processed) zurückgegriffen werden, mit mehr
    Kontext als beim ursprünglichen Tab-3-Durchlauf.

    Transkribiert+richtet NUR das übergebene Zeitfenster aus (nicht die
    ganze Datei — schnell), kappt die Wort-Zeitstempel wie gewohnt auf
    VAD-Intervalle (_clamp_words_to_vad), und sucht darin per
    _find_best_matching_span die Wortfolge, die dem erwarteten Text am
    nächsten kommt — statt wie beim ersten Durchlauf blind der Ausrichtung
    zu vertrauen.

    Schreibt NOCH NICHTS auf die Platte und verändert `entry` nicht — das
    übernimmt der Aufrufer (modules/reclip.py) erst nach erfolgreicher
    Round-Trip-Verifikation, mit der Möglichkeit, bei einem schlechteren
    Ergebnis auf das Original zurückzufallen.

    Gibt (erfolg, neuer_start_s, neuer_end_s, restfehler_wer) zurück, oder
    (False, None, None, None) bei einem Problem (Quelldatei fehlt, kein
    Fenster, keine Wörter erkannt)."""
    if pad_s is None:
        pad_s = settings.get("WORD_BOUNDARY_PAD_S")

    source_path = PROCESSED_DIR / entry.source_file
    if not source_path.exists():
        return False, None, None, None

    audio, sr = sf.read(str(source_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    total_duration = len(audio) / sr

    window_start = max(0.0, window_start)
    window_end = min(total_duration, window_end)
    if window_end <= window_start:
        return False, None, None, None

    window_audio = audio[int(window_start * sr):int(window_end * sr)]

    tmp_dir = PROCESSED_DIR / "_reclip_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{entry.id}_window.wav"
    try:
        sf.write(str(tmp_path), window_audio, sr, subtype="PCM_16")

        words = _transcribe_with_words(
            tmp_path, device=device, language=language,
            batch_size=int(settings.get("WHISPERX_BATCH_SIZE")),
        )
        if not words:
            return False, None, None, None

        vad_intervals = _get_vad_speech_intervals(tmp_path)
        words = _clamp_words_to_vad(words, vad_intervals)

        match = _find_best_matching_span(words, entry.text)
        if match is None:
            return False, None, None, None
        rel_start, rel_end, wer = match

        new_start = window_start + max(0.0, rel_start - pad_s)
        new_end = min(window_end, window_start + rel_end + pad_s)
        return True, new_start, new_end, wer
    finally:
        tmp_path.unlink(missing_ok=True)


def extract_clip_audio(source_path: Path, start_s: float, end_s: float) -> tuple:
    """Schneidet [start_s, end_s] aus der Quelldatei, resampled auf
    WORKING_SAMPLE_RATE falls nötig, wendet Rand-Trim + Fade an (gleiche
    Nachbearbeitung wie process_file()). Gibt (audio_float32, sr) zurück —
    schreibt noch nichts. Für modules/reclip.py (Neu-Zuschnitt bestehender
    Clips mit erweitertem Zeitfenster aus der Quelldatei)."""
    audio, sr = sf.read(str(source_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != WORKING_SAMPLE_RATE:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=WORKING_SAMPLE_RATE)
        sr = WORKING_SAMPLE_RATE

    start_sample = max(0, int(start_s * sr))
    end_sample = min(len(audio), int(end_s * sr))
    clip_audio = np.asarray(audio[start_sample:end_sample], dtype=np.float32)

    top_db = settings.get("SILENCE_TRIM_TOP_DB")
    max_trim_s = settings.get("SILENCE_TRIM_MAX_S")
    fade_s = settings.get("FADE_DURATION_S")
    clip_audio = _trim_and_fade(clip_audio, sr, top_db, max_trim_s, fade_s)
    return clip_audio, sr


# ---------------------------------------------------------------------
# Öffentliche Hauptfunktion
# ---------------------------------------------------------------------

def process_file(processed_path: Path, device: str = "cuda", language: str = WHISPERX_LANGUAGE) -> list[ClipEntry]:
    intervals = _get_vad_speech_intervals(processed_path)
    print(f"[segment] {processed_path.name}: {len(intervals)} VAD-Sprachintervall(e) gefunden")
    if not intervals:
        print(f"[segment] {processed_path.name}: keine Sprache erkannt -> 0 Clips")
        return []

    words = _transcribe_with_words(
        processed_path, device=device, language=language,
        batch_size=int(settings.get("WHISPERX_BATCH_SIZE")),
    )
    if not words:
        print(f"[segment] {processed_path.name}: whisperX lieferte keine Wörter -> 0 Clips")
        return []
    words = _clamp_words_to_vad(words, intervals)

    max_duration = settings.get("MAX_CLIP_DURATION_S")
    min_duration = settings.get("MIN_CLIP_DURATION_S")
    silence_gap = settings.get("MIN_SILENCE_GAP_S")
    pad_s = settings.get("WORD_BOUNDARY_PAD_S")

    clip_bounds = _build_clips_from_words(words, max_duration, min_duration, silence_gap, pad_s)
    print(f"[segment] {processed_path.name}: {len(clip_bounds)} Clip(s) aus Wort-Timestamps "
          f"gebaut (max. {max_duration}s, min. {min_duration}s)")
    if not clip_bounds:
        return []

    # Audio laden (voller Arbeitsstandard, nicht die 16kHz-VAD-Kopie)
    audio, sr = sf.read(str(processed_path))
    if sr != WORKING_SAMPLE_RATE:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=WORKING_SAMPLE_RATE)
        sr = WORKING_SAMPLE_RATE

    top_db = settings.get("SILENCE_TRIM_TOP_DB")
    max_trim_s = settings.get("SILENCE_TRIM_MAX_S")
    fade_s = settings.get("FADE_DURATION_S")

    entries = []
    for idx, (start, end) in enumerate(clip_bounds):
        start_sample = max(0, int(start * sr))
        end_sample = min(len(audio), int(end * sr))
        clip_audio = audio[start_sample:end_sample]
        clip_audio = _trim_and_fade(np.asarray(clip_audio, dtype=np.float32), sr, top_db, max_trim_s, fade_s)

        clip_id = f"{processed_path.stem}_{idx:03d}"
        clip_filename = f"{clip_id}.wav"
        clip_path = SEGMENTS_DIR / clip_filename
        sf.write(str(clip_path), clip_audio, sr, subtype="PCM_16")

        text = _text_for_range(words, start, end)

        entries.append(ClipEntry(
            id=clip_id,
            clip_filename=clip_filename,
            source_file=processed_path.name,
            start_s=round(start, 3),
            end_s=round(end, 3),
            duration_s=round(len(clip_audio) / sr, 3),
            text=text,
        ))

    return entries


def _release_whisperx():
    """whisperX-Modelle nach dem kompletten Tab-3-Lauf (alle Dateien) aus dem
    GPU-Speicher entfernen, statt sie dauerhaft resident zu halten.

    Grund: whisperX (besonders large-v3) ist selbst groß, läuft im GLEICHEN
    Prozess wie ein danach evtl. erneut genutztes Demucs (Tab 2) — ohne
    Freigabe würde ein dauerhaft gecachtes whisperX-Modell dort unnötig VRAM
    blockieren. Gleiches Prinzip wie bei Demucs/UVR (siehe
    modules/demucs_separation.py, separator_server/serve_separator.py)."""
    global _whisperx_model, _whisperx_align_model, _whisperx_align_metadata, _whisperx_language_loaded
    _whisperx_model = None
    _whisperx_align_model = None
    _whisperx_align_metadata = None
    _whisperx_language_loaded = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    try:
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
    finally:
        _release_whisperx()

    return all_entries, skipped


def save_manifest(entries: list[ClipEntry]):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump([asdict(e) for e in entries], fh, ensure_ascii=False, indent=2)


def load_manifest() -> list[ClipEntry]:
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    # Nur bekannte Felder übernehmen -> ein altes manifest.json mit
    # inzwischen entfernten Feldern (z.B. Umbenennungen bei Weiterentwicklung
    # der GUI) führt nicht zu einem Fehler beim Laden, sondern wird einfach
    # mit den aktuellen Feld-Defaults ergänzt.
    valid_fields = {f.name for f in dataclass_fields(ClipEntry)}
    return [ClipEntry(**{k: v for k, v in r.items() if k in valid_fields}) for r in raw]


if __name__ == "__main__":
    # Erlaubt, process_all() als EIGENSTÄNDIGEN Subprozess zu starten (siehe
    # app.py, handle_segment_and_transcribe). Grund: whisperX läuft intern
    # über CTranslate2 (faster-whisper) statt reinem PyTorch — dessen
    # GPU-Speicher wird von torch.cuda.empty_cache() NICHT erfasst, ein rein
    # python-seitiges "Referenz auf None setzen + gc.collect()" reicht hier
    # anders als bei Demucs nicht zuverlässig aus (siehe _release_whisperx()
    # oben, das half nur teilweise). Ein eigener Prozess garantiert dagegen
    # beim Beenden die vollständige Freigabe durch das Betriebssystem, egal
    # was CTranslate2/pyannote intern an Speicher gecacht halten.
    #
    # batch_size wird bewusst NICHT als CLI-Argument übergeben, sondern kommt
    # (wie beim In-Prozess-Aufruf auch) aus settings.get() zur Laufzeit —
    # der Subprozess liest dieselbe settings.json und bekommt so automatisch
    # den aktuellen Stand, ohne Extra-Parameter.
    import argparse

    parser = argparse.ArgumentParser(description="whisperX-Segmentierung als eigenständiger Prozess")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default=WHISPERX_LANGUAGE)
    args = parser.parse_args()

    def _cli_progress_cb(i, total, filename):
        # Eigenes, klar erkennbares Präfix, damit der Elternprozess (app.py)
        # diese Zeile zuverlässig von den normalen [segment]-Logmeldungen
        # unterscheiden kann.
        print(f"PDG_PROGRESS {i} {total} {filename}", flush=True)

    _entries, _skipped = process_all(device=args.device, language=args.language, progress_cb=_cli_progress_cb)
    print(f"PDG_DONE {len(_entries)} {json.dumps(_skipped, ensure_ascii=False)}", flush=True)
