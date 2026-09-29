"""
Automatisches Nachschneiden von Clips mit schlechtem Round-Trip-Score
("Kontrollschleife", Idee von Pete): für jeden Clip unterhalb der Round-
Trip-Warnschwelle wird versucht, ihn anhand eines erweiterten Zeitfensters
aus der Quelldatei neu zu schneiden (siehe
modules/segment_and_transcribe.reclip_entry), danach erneut per Round-Trip
verifiziert.

Sicherheitsnetz: Bringt der Neu-Zuschnitt keine Verbesserung (oder schlägt
fehl), wird auf den ORIGINAL-Clip zurückgefallen — kein Risiko, einen
ohnehin schon mangelhaften Clip noch schlechter zu machen. Der Clip bleibt
dann für die manuelle Kontrolle (Trimm-Tool in Tab 4) markiert.

Läuft als eigenständiger Subprozess (gleiches Prinzip wie
segment_and_transcribe.py/roundtrip_check.py) — GPU-Speicher wird beim
Beenden garantiert freigegeben.
"""
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SEGMENTS_DIR, PROCESSED_DIR, WHISPERX_LANGUAGE
from modules import settings
from modules.segment_and_transcribe import load_manifest, save_manifest, reclip_entry, extract_clip_audio
from modules.roundtrip_check import score_clip

import soundfile as sf


def reclip_all(device: str = "cuda", language: str = WHISPERX_LANGUAGE, progress_cb=None) -> tuple[int, int, int]:
    """Geht alle Clips unterhalb der Round-Trip-Warnschwelle durch und
    versucht, sie neu zu schneiden. Speichert nach JEDEM Clip (gleiches
    Prinzip wie Tab 3 seit 0.17.1).

    Gibt (verbessert, weiterhin_mangelhaft, uebersprungen) zurück."""
    entries = load_manifest()
    threshold = settings.get("ASR_CONFIDENCE_RED_THRESHOLD")
    margin = settings.get("RECLIP_MARGIN_S")
    max_attempts = int(settings.get("RECLIP_MAX_ATTEMPTS"))

    candidates = [e for e in entries if e.asr_confidence is not None and e.asr_confidence < threshold]
    total = len(candidates)
    improved, still_bad, skipped = 0, 0, 0

    for idx, entry in enumerate(candidates, start=1):
        clip_path = SEGMENTS_DIR / entry.clip_filename
        source_path = PROCESSED_DIR / entry.source_file
        original_score = entry.asr_confidence

        if not clip_path.exists() or not source_path.exists():
            skipped += 1
            if progress_cb:
                progress_cb(idx, total, entry.id)
            continue

        # Original sichern (im Speicher) -> garantierter Rückfall, falls
        # kein Versuch eine Verbesserung bringt.
        original_bytes = clip_path.read_bytes()
        original_start, original_end, original_duration = entry.start_s, entry.end_s, entry.duration_s

        best_result = None  # (score, audio, sr, new_start, new_end)
        for attempt in range(1, max_attempts + 1):
            window_margin = margin * attempt
            ok, new_start, new_end, _wer = reclip_entry(
                entry, entry.start_s - window_margin, entry.end_s + window_margin,
                device=device, language=language,
            )
            if not ok:
                continue

            candidate_audio, sr = extract_clip_audio(source_path, new_start, new_end)
            tmp_path = SEGMENTS_DIR / f"_reclip_candidate_{entry.id}.wav"
            try:
                sf.write(str(tmp_path), candidate_audio, sr, subtype="PCM_16")
                candidate_score = score_clip(tmp_path, entry.text, language=language)
            finally:
                tmp_path.unlink(missing_ok=True)

            if candidate_score is not None and (best_result is None or candidate_score > best_result[0]):
                best_result = (candidate_score, candidate_audio, sr, new_start, new_end)
            if best_result is not None and best_result[0] >= threshold:
                break  # gut genug -> weitere (teurere) Versuche mit größerem Fenster sparen

        if best_result is not None and best_result[0] > original_score:
            score, audio, sr, new_start, new_end = best_result
            sf.write(str(clip_path), audio, sr, subtype="PCM_16")
            entry.start_s = round(new_start, 3)
            entry.end_s = round(new_end, 3)
            entry.duration_s = round(len(audio) / sr, 3)
            entry.asr_confidence = score
            if score >= threshold:
                improved += 1
            else:
                still_bad += 1  # besser, aber noch nicht über der Schwelle
        else:
            # Keine Verbesserung -> Original unangetastet lassen (Datei war
            # ohnehin nicht verändert, nur zur Klarheit hier explizit)
            clip_path.write_bytes(original_bytes)
            entry.start_s, entry.end_s, entry.duration_s = original_start, original_end, original_duration
            entry.asr_confidence = original_score
            still_bad += 1

        save_manifest(entries)
        if progress_cb:
            progress_cb(idx, total, entry.id)

    reclip_tmp_dir = PROCESSED_DIR / "_reclip_tmp"
    if reclip_tmp_dir.exists():
        shutil.rmtree(reclip_tmp_dir, ignore_errors=True)

    return improved, still_bad, skipped


if __name__ == "__main__":
    # Eigenständiger Subprozess-Einstiegspunkt (siehe app.py,
    # handle_reclip) — gleiches Prinzip wie segment_and_transcribe.py seit
    # 0.20.7 / roundtrip_check.py seit 0.22.0: garantiert vollständige
    # GPU-Speicherfreigabe beim Beenden.
    def _cli_progress_cb(i, total, clip_id):
        print(f"PDG_PROGRESS {i} {total} {clip_id}", flush=True)

    _improved, _still_bad, _skipped = reclip_all(progress_cb=_cli_progress_cb)
    print(f"PDG_DONE {_improved} {_still_bad} {_skipped}", flush=True)
