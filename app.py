"""
Perfect Dataset GUI — Dataset-Prep-Tool für XTTS/RVC-Trainingsdaten.

Pipeline:
  1. Import & Vorverarbeitung        (fertig)
  2. Music/Noise Removal (Demucs htdemucs_ft + optional DeepFilterNet) (fertig)
  3. VAD-Segmentierung + Transkription (fertig)
  4. Review & Korrektur              (fertig)
  5. Export (XTTS + RVC)             (fertig)

GUI zweisprachig (Deutsch/Englisch, siehe i18n.py) — Deutsch ist Default.
"""
import json
import re
import subprocess
import sys
import pandas as pd
import numpy as np
import soundfile as sf
import gradio as gr

from pathlib import Path

from modules.audio_io import import_audio_files, list_raw_files
from modules.reset import reset_all
from modules.snapshot import create_snapshot, list_snapshots, restore_snapshot, delete_snapshot
from modules.demucs_separation import process_files as demucs_process_files, list_processed_files
from modules.uvr_separation import process_files as uvr_process_files
from modules.segment_and_transcribe import load_manifest, save_manifest
from modules.quality_score import score_all as nisqa_score_all
from modules import settings
from export.export_dataset import export_all, XTTS_DIR, RVC_DIR
from config import (
    WORKING_SAMPLE_RATE, EXPORT_SAMPLE_RATES, EXPORT_DIR, SEGMENTS_DIR, PROJECT_VERSION,
    SUPPORTED_TRANSCRIPTION_LANGUAGES, WHISPERX_LANGUAGE,
)
from i18n import t

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".opus"}
DEFAULT_LANG = "de"


def _lang_choices():
    return [("Deutsch", "de"), ("English", "en")]


def _input_lang_choices(lang):
    idx = 1 if lang == "de" else 2  # Index in SUPPORTED_TRANSCRIPTION_LANGUAGES-Tupel
    return [(row[idx], row[0]) for row in SUPPORTED_TRANSCRIPTION_LANGUAGES]


def Path_basename(p: str) -> str:
    return Path(p).name if p else "?"


def _raw_file_table():
    files = list_raw_files()
    return [[i + 1, f.name, False] for i, f in enumerate(files)]


def handle_raw_refresh():
    return _raw_file_table()


def handle_raw_save(table_data):
    files = list_raw_files()
    deleted = 0
    for row, f in zip(table_data, files):
        if bool(row[2]) and f.exists():
            f.unlink()
            deleted += 1
    msg = f"{deleted} gelöscht."
    return msg, _raw_file_table()


def handle_import(files, folder_files):
    combined = list(files or []) + list(folder_files or [])
    paths = [f.name for f in combined if Path(f.name).suffix.lower() in AUDIO_EXTENSIONS]
    if not paths:
        return "-", _raw_file_table()
    results = import_audio_files(paths)
    lines = []
    for r in results:
        if r.target_path:
            lines.append(
                f"✓ {Path_basename(r.source_path)} → {Path_basename(r.target_path)} "
                f"({r.original_sr} Hz → {r.target_sr} Hz, {r.duration_s:.1f}s)"
            )
        else:
            lines.append(f"✗ {Path_basename(r.source_path)}: {r.message}")
    return "\n".join(lines), _raw_file_table()


def _sanitize_filename_part(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE)
    return s.strip("_")


def _dataset_summary_data():
    entries = load_manifest()
    count = len(entries)
    if count == 0:
        return None
    durations = [e.duration_s for e in entries]
    total_s = sum(durations)
    scored = [e.nisqa_mos for e in entries if e.nisqa_mos is not None]
    avg_nisqa = sum(scored) / len(scored) if scored else None

    bins = [(0, 3), (3, 5), (5, 7), (7, 10), (10, float("inf"))]
    bin_labels = ["<3s", "3-5s", "5-7s", "7-10s", "≥10s"]
    bin_counts = [0] * len(bins)
    for d in durations:
        for i, (lo, hi) in enumerate(bins):
            if lo <= d < hi:
                bin_counts[i] += 1
                break

    return {
        "count": count, "total_s": total_s, "avg_s": total_s / count,
        "min_s": min(durations), "max_s": max(durations),
        "avg_nisqa": avg_nisqa, "scored_count": len(scored),
        "bin_labels": bin_labels, "bin_counts": bin_counts,
    }


def handle_dataset_summary_refresh(lang):
    data = _dataset_summary_data()
    empty_df = pd.DataFrame({"bucket": [], "count": []})
    if data is None:
        return t(lang, "dataset_summary_empty"), empty_df

    h = int(data["total_s"] // 3600)
    m = int((data["total_s"] % 3600) // 60)
    s = int(data["total_s"] % 60)
    nisqa_text = (
        f"{data['avg_nisqa']:.2f} ({data['scored_count']}/{data['count']} bewertet)"
        if data["avg_nisqa"] is not None else t(lang, "dataset_summary_nisqa_none")
    )
    text = t(
        lang, "dataset_summary_text",
        count=data["count"], hms=f"{h}:{m:02d}:{s:02d}",
        avg=f"{data['avg_s']:.1f}", min=f"{data['min_s']:.1f}", max=f"{data['max_s']:.1f}",
        nisqa=nisqa_text,
    )
    df = pd.DataFrame({"bucket": data["bin_labels"], "count": data["bin_counts"]})
    return text, df


def handle_export_run(speaker_name=""):
    summary = export_all()

    if summary.xtts_count == 0 and summary.rvc_count == 0:
        lines = [
            "⚠ Nichts exportiert — keine Clips vorhanden.",
            "Dafür müssen erst Tab 3 (Segmentierung + Transkription) und optional Tab 4 (Review) "
            "durchlaufen sein; Export liest ausschließlich aus den dort erzeugten Clips, nicht "
            "direkt aus Tab 2.",
        ]
        return "\n".join(lines), None

    lines = [
        f"✓ XTTS (LJSpeech, {EXPORT_SAMPLE_RATES['xtts']} Hz): {summary.xtts_count} → {XTTS_DIR}",
        f"✓ RVC ({EXPORT_SAMPLE_RATES['rvc']} Hz): {summary.rvc_count} → {RVC_DIR}",
    ]
    if summary.skipped_no_text:
        lines.append(f"⚠ {summary.skipped_no_text} clips ohne Text: nur RVC, nicht XTTS.")
    if summary.errors:
        lines.append(f"✗ {len(summary.errors)} Fehler:")
        lines.extend(f"   - {err}" for err in summary.errors)

    zip_path = None
    if summary.xtts_count or summary.rvc_count:
        import shutil
        import tempfile
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        speaker_part = _sanitize_filename_part(speaker_name)
        base_name = f"{speaker_part}_{timestamp}" if speaker_part else f"perfect_dataset_gui_export_{timestamp}"
        zip_base = str(Path(tempfile.gettempdir()) / base_name)
        shutil.make_archive(zip_base, "zip", root_dir=str(EXPORT_DIR), base_dir=".")
        zip_path = zip_base + ".zip"

    return "\n".join(lines), zip_path


def _processed_table_rows():
    files = list_processed_files()
    return [[i + 1, f.name, False] for i, f in enumerate(files)]


def handle_demucs_run(denoise_choice, music_mode, engine, progress=gr.Progress()):
    raw_files = list_raw_files()
    if not raw_files:
        return "-", _processed_table_rows()

    progress(0, desc="...")

    def _cb(i, total, filename):
        progress(i / total, desc=f"({i}/{total}) {filename}")

    process_fn = uvr_process_files if engine == "uvr" else demucs_process_files
    results = process_fn(
        raw_files, denoise=denoise_choice, music_mode=music_mode, progress_cb=_cb,
    )
    lines = []
    for r in results:
        if r.success:
            suffix = f" ({r.info})" if r.info else ""
            lines.append(f"✓ {Path_basename(r.source_path)} → {Path_basename(r.target_path)}{suffix}")
        else:
            lines.append(f"✗ {Path_basename(r.source_path)}: {r.message}")
    return "\n".join(lines), _processed_table_rows()


def handle_processed_refresh():
    return _processed_table_rows()


def handle_processed_save(table_data):
    files = list_processed_files()
    deleted = 0
    for row, f in zip(table_data, files):
        if bool(row[2]) and f.exists():
            f.unlink()
            deleted += 1
    msg = f"{deleted} gelöscht."
    return msg, _processed_table_rows()


SEGMENT_WORKER_PATH = Path(__file__).resolve().parent / "modules" / "segment_and_transcribe.py"


def handle_segment_and_transcribe(device, input_language, progress=gr.Progress()):
    progress(0, desc="...")

    # whisperX läuft als EIGENER Prozess (nicht mehr in-process) — Grund:
    # whisperX nutzt intern CTranslate2 (faster-whisper), dessen GPU-Speicher
    # sich mit torch.cuda.empty_cache() nicht zuverlässig freigeben lässt.
    # Ein eigener Prozess garantiert beim Beenden die vollständige Freigabe
    # durch das Betriebssystem, unabhängig davon, was CTranslate2/pyannote
    # intern an Speicher gecacht halten (siehe CHANGELOG 0.20.7).
    cmd = [sys.executable, str(SEGMENT_WORKER_PATH), "--device", device, "--language", input_language]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    entries_count = None
    skipped = []
    tail_lines = []
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("PDG_PROGRESS "):
                try:
                    _, i, total, filename = line.split(" ", 3)
                    progress(int(i) / int(total), desc=f"({i}/{total}) {filename}")
                except ValueError:
                    pass
            elif line.startswith("PDG_DONE "):
                rest = line[len("PDG_DONE "):]
                count_str, skipped_json = rest.split(" ", 1)
                entries_count = int(count_str)
                try:
                    skipped = json.loads(skipped_json)
                except Exception:
                    skipped = []
            else:
                print(line)  # [segment]-Meldungen & alles andere weiterhin im Terminal sichtbar
                tail_lines.append(line)
        proc.wait()
    except Exception:
        proc.kill()
        raise

    if proc.returncode != 0 or entries_count is None:
        tail = "\n".join(tail_lines[-15:])
        return (
            f"Fehler: Subprozess für Segmentierung/Transkription wurde mit Code "
            f"{proc.returncode} beendet, ohne ein Ergebnis zu melden.\n\n"
            f"Letzte Ausgabe:\n{tail}\n(voller Log im Terminal)"
        )

    entries = load_manifest()
    msg = f"{entries_count} clips."
    if skipped:
        msg += (
            f"\n\n⚠️ {len(skipped)} Datei(en) übersprungen (Fehler bei der Verarbeitung): "
            f"{', '.join(skipped)}. Das Manifest enthält trotzdem alle bis dahin bereits "
            "erfolgreich verarbeiteten Clips — nichts davon ist verloren. Details zum "
            "jeweiligen Fehler im Terminal-Log (Zeilen mit '[segment]')."
        )
    if not entries:
        msg += (
            "\n\nKeine Clips gefunden — keine Sprache erkannt, oder alle Abschnitte "
            "kürzer als die Mindest-Clip-Länge (Einstellungen-Tab). Details im Terminal-Log "
            "(Zeilen mit '[segment]')."
        )
    else:
        # NISQA direkt im Anschluss für alle Clips mitlaufen lassen, statt eines
        # separaten manuellen Klicks in Tab 4 — jetzt sogar mit noch mehr freiem
        # VRAM, weil der whisperX-Subprozess zu diesem Zeitpunkt schon komplett
        # beendet ist und seinen Speicher vollständig zurückgegeben hat.
        def _nisqa_cb(i, total, clip_id):
            progress(i / total, desc=f"NISQA-Bewertung ({i}/{total}) {clip_id}")

        scored, failed = nisqa_score_all(progress_cb=_nisqa_cb)
        msg += f"\n\n🎯 NISQA automatisch bewertet: {scored} Clip(s)."
        if failed:
            msg += f" {failed} fehlgeschlagen (NISQA-Server erreichbar? siehe start.sh)."

    return msg


def _score_badge(value, threshold, higher_is_better=True):
    if value is None:
        return '<span style="color:#888;">n/a</span>'
    bad = (value < threshold) if higher_is_better else (value > threshold)
    color = "#d33" if bad else "#2a7"
    return f'<span style="color:{color};font-weight:bold;">{value:.2f}</span>'


REVIEW_PLAY_COL = 3  # Spaltenindex von "▶" in der Review-Tabelle (0-basiert)


def _review_table_rows(entries):
    nisqa_threshold = settings.get("NISQA_MOS_RED_THRESHOLD")
    asr_threshold = settings.get("ASR_CONFIDENCE_RED_THRESHOLD")
    return [
        [i + 1, e.source_file, e.text, "▶",
         _score_badge(e.nisqa_mos, nisqa_threshold),
         _score_badge(e.asr_confidence, asr_threshold),
         False]
        for i, e in enumerate(entries)
    ]


def handle_review_play(evt: gr.SelectData, entries):
    """Spielt den Clip ab, wenn in der Review-Tabelle auf die ▶-Spalte
    geklickt wird. Bei Klick auf eine andere Spalte (z.B. Transkript zum
    Bearbeiten) bleibt der Player unverändert (gr.update()). Liefert
    zusätzlich den Clip-Dateinamen zurück (für handle_trim_apply — der Player
    selbst kennt nach dem Trimmen nur noch die neuen Audiodaten, nicht mehr,
    zu welcher Datei sie gehören)."""
    row, col = evt.index
    if col != REVIEW_PLAY_COL or row < 0 or row >= len(entries):
        return gr.update(), gr.update()
    clip_path = SEGMENTS_DIR / entries[row].clip_filename
    if not clip_path.exists():
        return gr.update(), gr.update()
    return str(clip_path), entries[row].clip_filename


def handle_trim_apply(audio_value, clip_filename, entries):
    """Speichert das im Player getrimmte Audio als neue Version der Clip-
    Datei. audio_value kommt vom gr.Audio mit type='numpy': (samplerate,
    numpy_array) nach Bestätigen des Trims in der Wellenform-Ansicht — als
    16-bit-Integer-Werte (-32768..32767), NICHT normalisiertes Float (siehe
    Gradio-Doku zu Audio.preprocess). Vor jeder Rechnung deshalb erst nach
    float32 im Bereich [-1, 1] umrechnen, wie der Rest des Projekts es auch
    handhabt (z.B. _trim_and_fade in segment_and_transcribe.py).

    Zusätzlich ein kurzer Fade-in/-out: Gradios Trim schneidet hart, ohne
    jede Überblendung — das erzeugt an einem ungünstigen Schnittpunkt
    (fernab eines Nulldurchgangs) ein hörbares Klick-/Clipping-Geräusch."""
    if not clip_filename or audio_value is None:
        return t(DEFAULT_LANG, "trim_status_none"), entries, _review_table_rows(entries)

    sr, audio_np = audio_value
    audio_np = np.asarray(audio_np)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)  # mehrkanalig -> auf Mono reduzieren
    audio_f32 = audio_np.astype(np.float32) / 32768.0

    fade_s = settings.get("FADE_DURATION_S")
    fade_len = min(int(fade_s * sr), len(audio_f32) // 2) if sr else 0
    if fade_len > 0:
        fade_in = np.linspace(0.0, 1.0, fade_len)
        fade_out = np.linspace(1.0, 0.0, fade_len)
        audio_f32[:fade_len] *= fade_in
        audio_f32[-fade_len:] *= fade_out

    clip_path = SEGMENTS_DIR / clip_filename
    sf.write(str(clip_path), audio_f32, sr, subtype="PCM_16")

    new_duration = len(audio_f32) / sr if sr else 0.0
    for e in entries:
        if e.clip_filename == clip_filename:
            e.duration_s = new_duration
            break
    save_manifest(entries)

    status = t(DEFAULT_LANG, "trim_status_saved", filename=clip_filename, duration=f"{new_duration:.2f}")
    return status, entries, _review_table_rows(entries)


def handle_review_refresh():
    entries = load_manifest()
    return entries, _review_table_rows(entries)


def handle_nisqa_score_all(progress=gr.Progress()):
    progress(0, desc="...")

    def _cb(i, total, clip_id):
        progress(i / total, desc=f"({i}/{total}) {clip_id}")

    scored, failed = nisqa_score_all(progress_cb=_cb)
    entries = load_manifest()
    msg = f"{scored} bewertet"
    if failed:
        msg += f", {failed} fehlgeschlagen (NISQA-Server erreichbar? siehe start.sh)"
    return entries, _review_table_rows(entries), msg


ROUNDTRIP_WORKER_PATH = Path(__file__).resolve().parent / "modules" / "roundtrip_check.py"


def handle_roundtrip_check(progress=gr.Progress()):
    """Round-Trip-Check als eigenständiger Subprozess (gleiches Prinzip wie
    handle_segment_and_transcribe seit 0.20.7) — garantiert vollständige
    GPU-Speicherfreigabe beim Beenden."""
    progress(0, desc="...")

    cmd = [sys.executable, str(ROUNDTRIP_WORKER_PATH)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    scored, failed = None, None
    tail_lines = []
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("PDG_PROGRESS "):
                try:
                    _, i, total, clip_id = line.split(" ", 3)
                    progress(int(i) / int(total), desc=f"({i}/{total}) {clip_id}")
                except ValueError:
                    pass
            elif line.startswith("PDG_DONE "):
                try:
                    _, scored_str, failed_str = line.split(" ", 2)
                    scored, failed = int(scored_str), int(failed_str)
                except ValueError:
                    pass
            else:
                print(line)
                tail_lines.append(line)
        proc.wait()
    except Exception:
        proc.kill()
        raise

    entries = load_manifest()
    if proc.returncode != 0 or scored is None:
        tail = "\n".join(tail_lines[-15:])
        msg = f"Fehler: Round-Trip-Subprozess mit Code {proc.returncode} beendet.\n\nLetzte Ausgabe:\n{tail}"
        return entries, _review_table_rows(entries), msg

    msg = f"{scored} geprüft"
    if failed:
        msg += f", {failed} fehlgeschlagen"
    return entries, _review_table_rows(entries), msg


RECLIP_WORKER_PATH = Path(__file__).resolve().parent / "modules" / "reclip.py"


def handle_reclip(progress=gr.Progress()):
    """Kontrollschleife (Pete-Idee) als eigenständiger Subprozess: schneidet
    Clips unterhalb der Round-Trip-Warnschwelle automatisch neu, mit
    Sicherheitsnetz (Rückfall auf Original bei keiner Verbesserung)."""
    progress(0, desc="...")

    cmd = [sys.executable, str(RECLIP_WORKER_PATH)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    improved, still_bad, skipped = None, None, None
    tail_lines = []
    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line.startswith("PDG_PROGRESS "):
                try:
                    _, i, total, clip_id = line.split(" ", 3)
                    progress(int(i) / int(total), desc=f"({i}/{total}) {clip_id}")
                except ValueError:
                    pass
            elif line.startswith("PDG_DONE "):
                try:
                    _, improved_str, still_bad_str, skipped_str = line.split(" ", 3)
                    improved, still_bad, skipped = int(improved_str), int(still_bad_str), int(skipped_str)
                except ValueError:
                    pass
            else:
                print(line)
                tail_lines.append(line)
        proc.wait()
    except Exception:
        proc.kill()
        raise

    entries = load_manifest()
    if proc.returncode != 0 or improved is None:
        tail = "\n".join(tail_lines[-15:])
        msg = f"Fehler: Neu-Zuschnitt-Subprozess mit Code {proc.returncode} beendet.\n\nLetzte Ausgabe:\n{tail}"
        return entries, _review_table_rows(entries), msg

    msg = f"{improved} Clip(s) verbessert, {still_bad} weiterhin unterhalb der Schwelle"
    if skipped:
        msg += f", {skipped} übersprungen (Datei fehlt)"
    return entries, _review_table_rows(entries), msg


def handle_review_save(table_data, entries):
    if not entries:
        return entries, _review_table_rows(entries), "-"

    # Gegen den AKTUELLEN Manifest-Stand mergen statt ihn blind zu überschreiben:
    # Falls zwischenzeitlich (z.B. durch einen erneuten Lauf von Tab 3, während
    # dieser Tab noch den alten Stand zeigt) neue Clips hinzugekommen sind,
    # dürfen die hier nicht verschwinden, nur weil sie nicht in der gerade
    # angezeigten Tabelle stehen.
    current_by_id = {e.id: e for e in load_manifest()}
    deleted = 0
    for row, e in zip(table_data, entries):
        if e.id not in current_by_id:
            continue  # zwischenzeitlich anderweitig entfernt -> nichts zu tun
        delete_flag = bool(row[6])
        if delete_flag:
            clip_path = SEGMENTS_DIR / e.clip_filename
            if clip_path.exists():
                clip_path.unlink()
            del current_by_id[e.id]
            deleted += 1
            continue
        current_by_id[e.id].text = str(row[2])

    kept = list(current_by_id.values())
    save_manifest(kept)
    msg = f"{deleted} gelöscht, {len(kept)} übrig."
    return kept, _review_table_rows(kept), msg


def handle_reset(confirm):
    if not confirm:
        no_change = (
            "-", _raw_file_table(), _processed_table_rows(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
        )
        return no_change
    summary = reset_all()
    return (
        summary, _raw_file_table(), _processed_table_rows(),
        "",           # Tab 3: Log leeren
        [], [],       # Tab 4: review_entries + review_table leeren
        "",           # Tab 4: review_status leeren
        "", None,     # Tab 5: Export-Log + Download leeren
    )


def handle_snapshot_create(label):
    name = create_snapshot(label)
    choices = list_snapshots()
    return name, gr.update(choices=choices, value=name)


def handle_snapshot_restore(name):
    if not name:
        return "-", _raw_file_table(), _processed_table_rows()
    msg = restore_snapshot(name)
    return msg, _raw_file_table(), _processed_table_rows()


def handle_snapshot_delete(name):
    if not name:
        return "-", gr.update(choices=list_snapshots(), value=None)
    msg = delete_snapshot(name)
    return msg, gr.update(choices=list_snapshots(), value=None)


# ---------------------------------------------------------------------
# UI aufbauen — jede übersetzbare Komponente wird in TR gesammelt, damit
# der Sprachwechsel-Handler am Ende alle auf einmal aktualisieren kann.
# ---------------------------------------------------------------------

TR = []  # Liste von (component, kind) — kind bestimmt, welches gr.update()-Feld befüllt wird


def reg(component, kind, key):
    TR.append((component, kind, key))
    return component


with gr.Blocks(title="Perfect Dataset GUI") as demo:
    gr.Markdown(f"# 🎙️ Perfect Dataset GUI `v{PROJECT_VERSION}`")
    subtitle_md = gr.Markdown(t(DEFAULT_LANG, "app_subtitle"))
    workstandard_md = gr.Markdown(t(
        DEFAULT_LANG, "workstandard_line",
        sr=WORKING_SAMPLE_RATE, xtts=EXPORT_SAMPLE_RATES["xtts"], rvc=EXPORT_SAMPLE_RATES["rvc"],
    ))
    reg(subtitle_md, "value", "app_subtitle"); reg(workstandard_md, "workstandard", None)

    with gr.Accordion(t(DEFAULT_LANG, "settings_title"), open=False) as settings_accordion:
        reg(settings_accordion, "label", "settings_title")
        language_dropdown = gr.Dropdown(
            choices=_lang_choices(), value=DEFAULT_LANG, label=t(DEFAULT_LANG, "language_label"),
        )
        reg(language_dropdown, "label", "language_label")

        snapshot_desc_md = gr.Markdown(t(DEFAULT_LANG, "settings_snapshot_desc"))
        reg(snapshot_desc_md, "value", "settings_snapshot_desc")
        with gr.Row():
            snapshot_label = gr.Textbox(
                label=t(DEFAULT_LANG, "snapshot_label_label"),
                placeholder=t(DEFAULT_LANG, "snapshot_label_placeholder"), scale=3,
            )
            reg(snapshot_label, "label", "snapshot_label_label"); reg(snapshot_label, "placeholder", "snapshot_label_placeholder")
            snapshot_create_btn = gr.Button(t(DEFAULT_LANG, "snapshot_create_btn"), size="sm", scale=1)
            reg(snapshot_create_btn, "value", "snapshot_create_btn")
        with gr.Row():
            snapshot_dropdown = gr.Dropdown(
                choices=list_snapshots(), label=t(DEFAULT_LANG, "snapshot_dropdown_label"), scale=3,
            )
            reg(snapshot_dropdown, "label", "snapshot_dropdown_label")
            snapshot_restore_btn = gr.Button(t(DEFAULT_LANG, "snapshot_restore_btn"), size="sm", scale=1)
            reg(snapshot_restore_btn, "value", "snapshot_restore_btn")
            snapshot_delete_btn = gr.Button(t(DEFAULT_LANG, "snapshot_delete_btn"), size="sm", scale=1)
            reg(snapshot_delete_btn, "value", "snapshot_delete_btn")
        snapshot_log = gr.Textbox(label=t(DEFAULT_LANG, "snapshot_log_label"), lines=2, interactive=False)
        reg(snapshot_log, "label", "snapshot_log_label")

        gr.Markdown("---")
        with gr.Row():
            reset_confirm = gr.Checkbox(label=t(DEFAULT_LANG, "reset_confirm_label"), scale=3)
            reg(reset_confirm, "label", "reset_confirm_label")
            reset_btn = gr.Button(t(DEFAULT_LANG, "reset_btn"), size="sm", variant="secondary", scale=1)
            reg(reset_btn, "value", "reset_btn")
        reset_log = gr.Textbox(label=t(DEFAULT_LANG, "reset_log_label"), lines=2, interactive=False)
        reg(reset_log, "label", "reset_log_label")

    with gr.Tabs():
        with gr.TabItem(t(DEFAULT_LANG, "tab1_title")) as tab1:
            reg(tab1, "label", "tab1_title")
            tab1_desc_md = gr.Markdown(t(DEFAULT_LANG, "tab1_desc", sr=WORKING_SAMPLE_RATE))
            reg(tab1_desc_md, "tab1_desc", None)
            with gr.Row():
                file_input = gr.File(
                    label=t(DEFAULT_LANG, "file_input_label"), file_count="multiple", file_types=["audio"],
                )
                reg(file_input, "label", "file_input_label")
                folder_input = gr.File(label=t(DEFAULT_LANG, "folder_input_label"), file_count="directory")
                reg(folder_input, "label", "folder_input_label")
            import_btn = gr.Button(t(DEFAULT_LANG, "import_btn"), variant="primary")
            reg(import_btn, "value", "import_btn")
            import_log = gr.Textbox(label=t(DEFAULT_LANG, "log_label"), lines=8, interactive=False)
            reg(import_log, "label", "log_label")
            with gr.Row():
                raw_refresh_btn = gr.Button(t(DEFAULT_LANG, "review_refresh_btn"), size="sm")
                reg(raw_refresh_btn, "value", "review_refresh_btn")
                raw_save_btn = gr.Button(t(DEFAULT_LANG, "review_save_btn"), size="sm", variant="primary")
                reg(raw_save_btn, "value", "review_save_btn")
            raw_table = gr.Dataframe(
                headers=[t(DEFAULT_LANG, "review_col_nr"), t(DEFAULT_LANG, "file_col"), t(DEFAULT_LANG, "review_col_delete")],
                label=t(DEFAULT_LANG, "raw_table_label"),
                datatype=["number", "str", "bool"],
                column_widths=["10%", "80%", "10%"],
                type="array", interactive=True,
                value=_raw_file_table(),
            )
            reg(raw_table, "label", "raw_table_label"); reg(raw_table, "headers3", None)
            import_btn.click(fn=handle_import, inputs=[file_input, folder_input], outputs=[import_log, raw_table])
            raw_refresh_btn.click(fn=handle_raw_refresh, inputs=[], outputs=[raw_table])
            raw_save_btn.click(fn=handle_raw_save, inputs=[raw_table], outputs=[import_log, raw_table])

        with gr.TabItem(t(DEFAULT_LANG, "tab2_title")) as tab2:
            reg(tab2, "label", "tab2_title")
            tab2_desc_md = gr.Markdown(t(DEFAULT_LANG, "tab2_desc"))
            reg(tab2_desc_md, "value", "tab2_desc")
            engine_radio = gr.Radio(
                [(t(DEFAULT_LANG, "engine_opt_demucs"), "demucs"),
                 (t(DEFAULT_LANG, "engine_opt_uvr"), "uvr")],
                value="demucs", label=t(DEFAULT_LANG, "engine_label"),
                info=t(DEFAULT_LANG, "engine_info"),
            )
            reg(engine_radio, "label", "engine_label"); reg(engine_radio, "info", "engine_info"); reg(engine_radio, "engine_choices", None)
            music_mode_radio = gr.Radio(
                [(t(DEFAULT_LANG, "music_mode_opt1"), "remove_music"),
                 (t(DEFAULT_LANG, "music_mode_opt2"), "cut_music_sections")],
                value="remove_music", label=t(DEFAULT_LANG, "music_mode_label"),
                info=t(DEFAULT_LANG, "music_mode_info"),
            )
            reg(music_mode_radio, "label", "music_mode_label"); reg(music_mode_radio, "info", "music_mode_info"); reg(music_mode_radio, "music_choices", None)
            denoise_checkbox = gr.Checkbox(value=True, label=t(DEFAULT_LANG, "denoise_checkbox_label"))
            reg(denoise_checkbox, "label", "denoise_checkbox_label")
            demucs_btn = gr.Button(t(DEFAULT_LANG, "demucs_btn"), variant="primary")
            reg(demucs_btn, "value", "demucs_btn")
            demucs_log = gr.Textbox(label=t(DEFAULT_LANG, "log_label"), lines=8, interactive=False)
            reg(demucs_log, "label", "log_label")
            with gr.Row():
                processed_refresh_btn = gr.Button(t(DEFAULT_LANG, "review_refresh_btn"), size="sm")
                reg(processed_refresh_btn, "value", "review_refresh_btn")
                processed_save_btn = gr.Button(t(DEFAULT_LANG, "review_save_btn"), size="sm", variant="primary")
                reg(processed_save_btn, "value", "review_save_btn")
            processed_table = gr.Dataframe(
                headers=[t(DEFAULT_LANG, "review_col_nr"), t(DEFAULT_LANG, "file_col"), t(DEFAULT_LANG, "review_col_delete")],
                label=t(DEFAULT_LANG, "processed_table_label"),
                datatype=["number", "str", "bool"],
                column_widths=["10%", "80%", "10%"],
                type="array", interactive=True,
                value=_processed_table_rows(),
            )
            reg(processed_table, "label", "processed_table_label"); reg(processed_table, "headers3", None)
            demucs_btn.click(
                fn=handle_demucs_run, inputs=[denoise_checkbox, music_mode_radio, engine_radio],
                outputs=[demucs_log, processed_table],
            )
            processed_refresh_btn.click(fn=handle_processed_refresh, inputs=[], outputs=[processed_table])
            processed_save_btn.click(fn=handle_processed_save, inputs=[processed_table], outputs=[demucs_log, processed_table])

        with gr.TabItem(t(DEFAULT_LANG, "tab3_title")) as tab3:
            reg(tab3, "label", "tab3_title")
            tab3_desc_md = gr.Markdown(t(DEFAULT_LANG, "tab3_desc"))
            reg(tab3_desc_md, "value", "tab3_desc")
            input_language_dropdown = gr.Dropdown(
                choices=_input_lang_choices(DEFAULT_LANG), value=WHISPERX_LANGUAGE,
                label=t(DEFAULT_LANG, "input_language_label"),
            )
            reg(input_language_dropdown, "input_lang", None)
            device_choice = gr.Radio(
                [("GPU (CUDA)", "cuda"), ("CPU", "cpu")], value="cuda",
                label=t(DEFAULT_LANG, "device_choice_label"),
            )
            reg(device_choice, "label", "device_choice_label")
            segment_btn = gr.Button(t(DEFAULT_LANG, "segment_btn"), variant="primary")
            reg(segment_btn, "value", "segment_btn")
            segment_log = gr.Textbox(label=t(DEFAULT_LANG, "log_label"), lines=4, interactive=False)
            reg(segment_log, "label", "log_label")
            segment_btn.click(
                fn=handle_segment_and_transcribe, inputs=[device_choice, input_language_dropdown],
                outputs=[segment_log],
            )

        with gr.TabItem(t(DEFAULT_LANG, "tab4_title")) as tab4:
            reg(tab4, "label", "tab4_title")
            tab4_desc_md = gr.Markdown(t(
                DEFAULT_LANG, "tab4_desc",
                nisqa_threshold=settings.get("NISQA_MOS_RED_THRESHOLD"),
                asr_threshold=settings.get("ASR_CONFIDENCE_RED_THRESHOLD"),
            ))
            reg(tab4_desc_md, "tab4_desc", None)
            review_entries = gr.State([])

            with gr.Row():
                review_refresh_btn = gr.Button(t(DEFAULT_LANG, "review_refresh_btn"), size="sm")
                reg(review_refresh_btn, "value", "review_refresh_btn")
                nisqa_score_btn = gr.Button(t(DEFAULT_LANG, "nisqa_score_btn"), size="sm")
                reg(nisqa_score_btn, "value", "nisqa_score_btn")
                roundtrip_btn = gr.Button(t(DEFAULT_LANG, "roundtrip_btn"), size="sm")
                reg(roundtrip_btn, "value", "roundtrip_btn")
                review_save_btn = gr.Button(t(DEFAULT_LANG, "review_save_btn"), variant="primary", size="sm")
                reg(review_save_btn, "value", "review_save_btn")
            with gr.Row():
                reclip_btn = gr.Button(t(DEFAULT_LANG, "reclip_btn"), size="sm")
                reg(reclip_btn, "value", "reclip_btn")

            review_status = gr.Markdown("")
            selected_clip_filename = gr.State(None)
            review_audio_player = gr.Audio(
                label=t(DEFAULT_LANG, "review_audio_label"), interactive=True, autoplay=True,
            )
            reg(review_audio_player, "label", "review_audio_label")
            with gr.Row():
                trim_apply_btn = gr.Button(t(DEFAULT_LANG, "trim_apply_btn"), size="sm")
                reg(trim_apply_btn, "value", "trim_apply_btn")
                trim_status = gr.Markdown("")
            review_table = gr.Dataframe(
                headers=[
                    t(DEFAULT_LANG, "review_col_nr"), t(DEFAULT_LANG, "review_col_source"),
                    t(DEFAULT_LANG, "review_col_transcript"), t(DEFAULT_LANG, "review_col_play"),
                    t(DEFAULT_LANG, "review_col_nisqa"), t(DEFAULT_LANG, "review_col_whisper"),
                    t(DEFAULT_LANG, "review_col_delete"),
                ],
                datatype=["number", "str", "str", "str", "markdown", "markdown", "bool"],
                column_widths=["5%", "12%", "38%", "7%", "12%", "12%", "7%"],
                type="array", interactive=True, wrap=True, value=[],
            )
            reg(review_table, "headers6", None)

            review_refresh_btn.click(fn=handle_review_refresh, inputs=[], outputs=[review_entries, review_table])
            nisqa_score_btn.click(
                fn=handle_nisqa_score_all, inputs=[], outputs=[review_entries, review_table, review_status],
            )
            roundtrip_btn.click(
                fn=handle_roundtrip_check, inputs=[], outputs=[review_entries, review_table, review_status],
            )
            reclip_btn.click(
                fn=handle_reclip, inputs=[], outputs=[review_entries, review_table, review_status],
            )
            review_save_btn.click(
                fn=handle_review_save, inputs=[review_table, review_entries],
                outputs=[review_entries, review_table, review_status],
            )
            review_table.select(
                # WICHTIG: handle_review_play muss DIREKT am select()-Ereignis
                # hängen, nicht über .then() verkettet — Gradio gibt die
                # Klick-Positionsdaten (SelectData, welche Zeile/Spalte) nur an
                # die direkt gebundene Funktion weiter. Ein vorgeschalteter
                # .then()-Schritt (Versuch aus 0.22.1 gegen klebende Trimm-
                # Grenzen beim Clip-Wechsel) bekam dadurch bei jedem Klick
                # `None` statt der Klickposition und crashte — Wiedergabe war
                # komplett kaputt. Der Trimm-Grenzen-Bug ist damit wieder
                # ungelöst, aber Abspielen funktioniert wieder zuverlässig.
                fn=handle_review_play, inputs=[review_entries],
                outputs=[review_audio_player, selected_clip_filename],
            )
            trim_apply_btn.click(
                fn=handle_trim_apply, inputs=[review_audio_player, selected_clip_filename, review_entries],
                outputs=[trim_status, review_entries, review_table],
            )

        with gr.TabItem(t(DEFAULT_LANG, "tab5_title")) as tab5:
            reg(tab5, "label", "tab5_title")
            tab5_desc_md = gr.Markdown(t(
                DEFAULT_LANG, "tab5_desc",
                xtts=EXPORT_SAMPLE_RATES["xtts"], rvc=EXPORT_SAMPLE_RATES["rvc"],
                lufs=settings.get("TARGET_LUFS"),
            ))
            reg(tab5_desc_md, "tab5_desc", None)

            dataset_summary_title_md = gr.Markdown(f"### {t(DEFAULT_LANG, 'dataset_summary_title')}")
            reg(dataset_summary_title_md, "dataset_summary_title_header", None)
            dataset_summary_refresh_btn = gr.Button(t(DEFAULT_LANG, "dataset_summary_refresh_btn"), size="sm")
            reg(dataset_summary_refresh_btn, "value", "dataset_summary_refresh_btn")
            dataset_summary_md = gr.Markdown(t(DEFAULT_LANG, "dataset_summary_empty"))
            dataset_summary_chart = gr.BarPlot(
                value=pd.DataFrame({"bucket": [], "count": []}), x="bucket", y="count",
                x_title=t(DEFAULT_LANG, "dataset_summary_chart_x"),
                y_title=t(DEFAULT_LANG, "dataset_summary_chart_y"),
                label=t(DEFAULT_LANG, "dataset_summary_chart_label"), height=220,
            )
            reg(dataset_summary_chart, "label", "dataset_summary_chart_label")
            dataset_summary_refresh_btn.click(
                fn=handle_dataset_summary_refresh, inputs=[language_dropdown],
                outputs=[dataset_summary_md, dataset_summary_chart],
            )

            speaker_name_input = gr.Textbox(
                label=t(DEFAULT_LANG, "speaker_name_label"),
                placeholder=t(DEFAULT_LANG, "speaker_name_placeholder"),
            )
            reg(speaker_name_input, "label", "speaker_name_label")
            reg(speaker_name_input, "placeholder", "speaker_name_placeholder")

            export_btn = gr.Button(t(DEFAULT_LANG, "export_btn"), variant="primary")
            reg(export_btn, "value", "export_btn")
            export_log = gr.Textbox(label=t(DEFAULT_LANG, "log_label"), lines=8, interactive=False)
            reg(export_log, "label", "log_label")
            export_download = gr.File(label=t(DEFAULT_LANG, "export_download_label"))
            reg(export_download, "label", "export_download_label")
            export_btn.click(
                fn=handle_export_run, inputs=[speaker_name_input], outputs=[export_log, export_download],
            )

        with gr.TabItem(t(DEFAULT_LANG, "tab_settings_title")) as tab_settings:
            reg(tab_settings, "label", "tab_settings_title")
            settings_desc_md = gr.Markdown(t(DEFAULT_LANG, "settings_desc"))
            reg(settings_desc_md, "value", "settings_desc")

            SETTINGS_ORDER = list(settings.SETTINGS_SPEC.keys())
            settings_inputs = {}
            settings_group_headers = {}
            SETTINGS_GROUPS = ["segmentation", "music", "quality", "export"]
            grouped_names = {g: [] for g in SETTINGS_GROUPS}
            for _name, _spec in settings.SETTINGS_SPEC.items():
                grouped_names[_spec[4]].append(_name)

            for group in SETTINGS_GROUPS:
                header = gr.Markdown(f"### {t(DEFAULT_LANG, f'settings_group_{group}')}")
                settings_group_headers[group] = header
                reg(header, "settings_group_header", group)
                with gr.Row():
                    for name in grouped_names[group]:
                        default, min_v, max_v, step, _grp, precision = settings.SETTINGS_SPEC[name]
                        comp = gr.Number(
                            value=settings.get(name),
                            label=t(DEFAULT_LANG, f"setting_{name}_label"),
                            info=t(DEFAULT_LANG, f"setting_{name}_info"),
                            minimum=min_v, maximum=max_v, step=step, precision=precision,
                        )
                        reg(comp, "setting_label_info", name)
                        settings_inputs[name] = comp

            with gr.Row():
                settings_save_btn = gr.Button(t(DEFAULT_LANG, "settings_save_btn"), variant="primary")
                reg(settings_save_btn, "value", "settings_save_btn")
                settings_reset_btn = gr.Button(t(DEFAULT_LANG, "settings_reset_btn"))
                reg(settings_reset_btn, "value", "settings_reset_btn")
            settings_status = gr.Markdown("")

            def handle_settings_save(lang, *values):
                settings.set_values(dict(zip(SETTINGS_ORDER, values)))
                return (
                    t(lang, "settings_status_saved"),
                    t(lang, "tab4_desc",
                      nisqa_threshold=settings.get("NISQA_MOS_RED_THRESHOLD"),
                      asr_threshold=settings.get("ASR_CONFIDENCE_RED_THRESHOLD")),
                    t(lang, "tab5_desc",
                      xtts=EXPORT_SAMPLE_RATES["xtts"], rvc=EXPORT_SAMPLE_RATES["rvc"],
                      lufs=settings.get("TARGET_LUFS")),
                )

            def handle_settings_reset(lang):
                settings.reset_all()
                values = [settings.get(name) for name in SETTINGS_ORDER]
                return (
                    t(lang, "settings_status_reset"),
                    t(lang, "tab4_desc",
                      nisqa_threshold=settings.get("NISQA_MOS_RED_THRESHOLD"),
                      asr_threshold=settings.get("ASR_CONFIDENCE_RED_THRESHOLD")),
                    t(lang, "tab5_desc",
                      xtts=EXPORT_SAMPLE_RATES["xtts"], rvc=EXPORT_SAMPLE_RATES["rvc"],
                      lufs=settings.get("TARGET_LUFS")),
                    *values,
                )

            settings_save_btn.click(
                fn=handle_settings_save,
                inputs=[language_dropdown] + [settings_inputs[n] for n in SETTINGS_ORDER],
                outputs=[settings_status, tab4_desc_md, tab5_desc_md],
            )
            settings_reset_btn.click(
                fn=handle_settings_reset, inputs=[language_dropdown],
                outputs=[settings_status, tab4_desc_md, tab5_desc_md] + [settings_inputs[n] for n in SETTINGS_ORDER],
            )

    reset_btn.click(
        fn=handle_reset, inputs=[reset_confirm],
        outputs=[
            reset_log, raw_table, processed_table,
            segment_log,
            review_entries, review_table, review_status,
            export_log, export_download,
        ],
    )
    snapshot_create_btn.click(
        fn=handle_snapshot_create, inputs=[snapshot_label], outputs=[snapshot_log, snapshot_dropdown],
    )
    snapshot_restore_btn.click(
        fn=handle_snapshot_restore, inputs=[snapshot_dropdown], outputs=[snapshot_log, raw_table, processed_table],
    )
    snapshot_delete_btn.click(
        fn=handle_snapshot_delete, inputs=[snapshot_dropdown], outputs=[snapshot_log, snapshot_dropdown],
    )

    # -------------------------------------------------------------
    # Sprachwechsel: EIN Handler aktualisiert alle registrierten
    # Komponenten aus TR in der Reihenfolge ihrer Registrierung.
    # -------------------------------------------------------------
    def handle_language_change(lang):
        updates = []
        for component, kind, key in TR:
            if kind == "value":
                updates.append(gr.update(value=t(lang, key)))
            elif kind == "label":
                updates.append(gr.update(label=t(lang, key)))
            elif kind == "placeholder":
                updates.append(gr.update(placeholder=t(lang, key)))
            elif kind == "info":
                updates.append(gr.update(info=t(lang, key)))
            elif kind == "headers3":
                updates.append(gr.update(headers=[
                    t(lang, "review_col_nr"), t(lang, "file_col"), t(lang, "review_col_delete"),
                ]))
            elif kind == "headers6":
                updates.append(gr.update(headers=[
                    t(lang, "review_col_nr"), t(lang, "review_col_source"), t(lang, "review_col_transcript"),
                    t(lang, "review_col_play"), t(lang, "review_col_nisqa"), t(lang, "review_col_whisper"),
                    t(lang, "review_col_delete"),
                ]))
            elif kind == "music_choices":
                updates.append(gr.update(choices=[
                    (t(lang, "music_mode_opt1"), "remove_music"),
                    (t(lang, "music_mode_opt2"), "cut_music_sections"),
                ]))
            elif kind == "engine_choices":
                updates.append(gr.update(choices=[
                    (t(lang, "engine_opt_demucs"), "demucs"),
                    (t(lang, "engine_opt_uvr"), "uvr"),
                ]))
            elif kind == "input_lang":
                updates.append(gr.update(choices=_input_lang_choices(lang), label=t(lang, "input_language_label")))
            elif kind == "workstandard":
                updates.append(gr.update(value=t(
                    lang, "workstandard_line",
                    sr=WORKING_SAMPLE_RATE, xtts=EXPORT_SAMPLE_RATES["xtts"], rvc=EXPORT_SAMPLE_RATES["rvc"],
                )))
            elif kind == "tab1_desc":
                updates.append(gr.update(value=t(lang, "tab1_desc", sr=WORKING_SAMPLE_RATE)))
            elif kind == "tab4_desc":
                updates.append(gr.update(value=t(
                    lang, "tab4_desc",
                    nisqa_threshold=settings.get("NISQA_MOS_RED_THRESHOLD"),
                    asr_threshold=settings.get("ASR_CONFIDENCE_RED_THRESHOLD"),
                )))
            elif kind == "tab5_desc":
                updates.append(gr.update(value=t(
                    lang, "tab5_desc",
                    xtts=EXPORT_SAMPLE_RATES["xtts"], rvc=EXPORT_SAMPLE_RATES["rvc"],
                    lufs=settings.get("TARGET_LUFS"),
                )))
            elif kind == "settings_group_header":
                updates.append(gr.update(value=f"### {t(lang, f'settings_group_{key}')}"))
            elif kind == "dataset_summary_title_header":
                updates.append(gr.update(value=f"### {t(lang, 'dataset_summary_title')}"))
            elif kind == "setting_label_info":
                updates.append(gr.update(
                    label=t(lang, f"setting_{key}_label"), info=t(lang, f"setting_{key}_info"),
                ))
        return updates

    language_dropdown.change(fn=handle_language_change, inputs=[language_dropdown], outputs=[c for c, _, _ in TR])


if __name__ == "__main__":
    import atexit
    import signal
    import sys

    _cleanup_done = {"done": False}

    def _reset_on_exit():
        # Läuft potenziell über mehrere Wege (atexit UND Signal-Handler) —
        # Flag verhindert doppeltes Ausführen/doppelte Log-Ausgabe.
        if _cleanup_done["done"]:
            return
        _cleanup_done["done"] = True
        print("\nGUI wird beendet — setze alle Zwischenergebnisse zurück...")
        print(reset_all())

    def _signal_handler(signum, frame):
        # atexit allein reicht nicht zuverlässig: Gradio/Uvicorn beenden den
        # Prozess bei SIGINT (Strg+C) teils über einen schnellen Exit-Pfad,
        # der atexit-Hooks überspringt; bei SIGTERM (Terminal-/Fenster
        # schließen, kill) greift atexit standardmäßig gar nicht. Deshalb
        # hier explizit für beide Signale zuerst aufräumen, dann beenden.
        _reset_on_exit()
        sys.exit(0)

    atexit.register(_reset_on_exit)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    demo.launch()
