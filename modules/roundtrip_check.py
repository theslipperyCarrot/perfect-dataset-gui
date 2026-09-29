"""
Round-Trip-Check für Tab 4 (Review): transkribiert jeden fertigen Clip noch
einmal, komplett unabhängig von der ursprünglichen Segmentierung (kein
gemeinsamer Kontext, keine Wort-Alignment-Daten), und vergleicht das Ergebnis
per Wort-Fehlerrate (WER) mit dem zugewiesenen Text.

Deckt genau die Fehlerklasse auf, die ein reiner Alignment-Konfidenzwert
verschluckt: ein Clip, dem ein Nachbarwort/eine Pause mit ins Audio gerutscht
ist, hat einen korrekt erkannten Text für die eigentlichen Wörter (hohe
Alignment-Konfidenz möglich) — aber die isolierte Neu-Transkription des
tatsächlichen Audios enthält dann ein Wort zu viel/zu wenig, was die WER
deutlich sichtbar macht.

Läuft als EIGENSTÄNDIGER Subprozess (siehe CLI-Einstiegspunkt unten, analog
zu segment_and_transcribe.py seit 0.20.7) — GPU-Speicher wird beim Beenden
garantiert freigegeben, unabhängig davon was faster-whisper/CTranslate2
intern an Speicher gecacht halten.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SEGMENTS_DIR, ROUNDTRIP_MODEL, ROUNDTRIP_COMPUTE_TYPE, WHISPERX_LANGUAGE
from modules.segment_and_transcribe import load_manifest, save_manifest
from modules.torchaudio_compat import ensure_patched as ensure_torchaudio_compat
from modules.text_compare import normalize_word, word_error_rate

_model = None


def _load_model():
    global _model
    if _model is None:
        ensure_torchaudio_compat()
        from faster_whisper import WhisperModel
        _model = WhisperModel(ROUNDTRIP_MODEL, device="cuda", compute_type=ROUNDTRIP_COMPUTE_TYPE)
    return _model


def score_clip(clip_path: Path, expected_text: str, language: str = WHISPERX_LANGUAGE) -> float:
    """Gibt einen Score 0-1 zurück (1 = perfekte Übereinstimmung), oder None
    bei Fehler (Datei fehlt, Transkription schlägt fehl).

    Zusätzlich zum WER-basierten Gesamtscore ein härterer, gezielter Check
    für das ERSTE und LETZTE Wort: ein einzelnes fehlendes/zusätzliches Wort
    am Rand senkt den reinen WER-Durchschnitt bei einem langen Satz kaum
    merklich ab (1 von 15 Wörtern -> nur -0.07) — genau das ist aber die
    Fehlerklasse, die die Clip-Grenzen betrifft (siehe Praxisbeispiel:
    fehlendes "Die" am Anfang, WER-Score allein 0.93, deutlich zu hoch). Bei
    einem Rand-Wort-Mismatch wird der Score deshalb zusätzlich gedeckelt."""
    if not clip_path.exists():
        return None
    try:
        model = _load_model()
        segments, _info = model.transcribe(str(clip_path), language=language, beam_size=1)
        roundtrip_text = " ".join(seg.text.strip() for seg in segments).strip()

        wer = word_error_rate(expected_text, roundtrip_text)
        score = max(0.0, 1.0 - wer)

        ref_words = [w for w in (normalize_word(w) for w in expected_text.split()) if w]
        hyp_words = [w for w in (normalize_word(w) for w in roundtrip_text.split()) if w]
        if ref_words and not hyp_words:
            score = 0.0
        elif ref_words and hyp_words:
            boundary_mismatch = ref_words[0] != hyp_words[0] or ref_words[-1] != hyp_words[-1]
            if boundary_mismatch:
                score = min(score, 0.5)

        return round(score, 3)
    except Exception as e:
        print(f"[roundtrip] Fehler bei {clip_path.name}: {e}")
        return None


def score_all(progress_cb=None) -> tuple[int, int]:
    """Prüft alle Clips im Manifest. Gibt (geprüft, fehlgeschlagen) zurück.
    Speichert nach JEDEM Clip (gleiches Prinzip wie Tab 3 seit 0.17.1) —
    ein Abbruch mittendrin verliert nicht die bereits geprüften Clips."""
    entries = load_manifest()
    scored, failed = 0, 0
    total = len(entries)
    for i, e in enumerate(entries, start=1):
        score = score_clip(SEGMENTS_DIR / e.clip_filename, e.text)
        if score is not None:
            e.asr_confidence = score
            scored += 1
        else:
            failed += 1
        save_manifest(entries)
        if progress_cb:
            progress_cb(i, total, e.id)
    return scored, failed


if __name__ == "__main__":
    # Eigenständiger Subprozess-Einstiegspunkt (siehe app.py,
    # handle_roundtrip_check) — gleiches Prinzip wie
    # modules/segment_and_transcribe.py seit 0.20.7: garantiert vollständige
    # GPU-Speicherfreigabe beim Beenden, unabhängig davon was
    # faster-whisper/CTranslate2 intern gecacht halten.
    def _cli_progress_cb(i, total, clip_id):
        print(f"PDG_PROGRESS {i} {total} {clip_id}", flush=True)

    _scored, _failed = score_all(progress_cb=_cli_progress_cb)
    print(f"PDG_DONE {_scored} {_failed}", flush=True)
