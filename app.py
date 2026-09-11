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
import gradio as gr

from pathlib import Path

from modules.audio_io import import_audio_files, list_raw_files
from modules.reset import reset_all
from modules.snapshot import create_snapshot, list_snapshots, restore_snapshot, delete_snapshot
from modules.demucs_separation import process_files as demucs_process_files, list_processed_files
from modules.segment_and_transcribe import process_all as segment_and_transcribe_all, load_manifest, save_manifest
from modules.quality_score import score_all as nisqa_score_all
from export.export_dataset import export_all, XTTS_DIR, RVC_DIR
from config import (
    WORKING_SAMPLE_RATE, EXPORT_SAMPLE_RATES, EXPORT_DIR, SEGMENTS_DIR, TARGET_LUFS,
    NISQA_MOS_RED_THRESHOLD, ASR_CONFIDENCE_RED_THRESHOLD, PROJECT_VERSION,
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


def handle_export_run():
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
        zip_base = str(Path(tempfile.gettempdir()) / f"perfect_dataset_gui_export_{timestamp}")
        shutil.make_archive(zip_base, "zip", root_dir=str(EXPORT_DIR), base_dir=".")
        zip_path = zip_base + ".zip"

    return "\n".join(lines), zip_path


def _processed_table_rows():
    files = list_processed_files()
    return [[i + 1, f.name, False] for i, f in enumerate(files)]


def handle_demucs_run(denoise_choice, music_mode, progress=gr.Progress()):
    raw_files = list_raw_files()
    if not raw_files:
        return "-", _processed_table_rows()

    progress(0, desc="...")

    def _cb(i, total, filename):
        progress(i / total, desc=f"({i}/{total}) {filename}")

    results = demucs_process_files(
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


def handle_segment_and_transcribe(device, input_language, progress=gr.Progress()):
    progress(0, desc="...")

    def _cb(i, total, filename):
        progress(i / total, desc=f"({i}/{total}) {filename}")

    try:
        entries, skipped = segment_and_transcribe_all(device=device, language=input_language, progress_cb=_cb)
        msg = f"{len(entries)} clips."
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
                "kürzer als MIN_CLIP_DURATION_S (config.py). Details im Terminal-Log "
                "(Zeilen mit '[segment]')."
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Fehler: {e}\n(voller Traceback im Terminal-Log)"
    return msg


def _score_badge(value, threshold, higher_is_better=True):
    if value is None:
        return '<span style="color:#888;">n/a</span>'
    bad = (value < threshold) if higher_is_better else (value > threshold)
    color = "#d33" if bad else "#2a7"
    return f'<span style="color:{color};font-weight:bold;">{value:.2f}</span>'


REVIEW_PLAY_COL = 3  # Spaltenindex von "▶" in der Review-Tabelle (0-basiert)


def _review_table_rows(entries):
    return [
        [i + 1, e.source_file, e.text, "▶",
         _score_badge(e.nisqa_mos, NISQA_MOS_RED_THRESHOLD),
         _score_badge(e.asr_confidence, ASR_CONFIDENCE_RED_THRESHOLD),
         False]
        for i, e in enumerate(entries)
    ]


def handle_review_play(evt: gr.SelectData, entries):
    """Spielt den Clip ab, wenn in der Review-Tabelle auf die ▶-Spalte
    geklickt wird. Bei Klick auf eine andere Spalte (z.B. Transkript zum
    Bearbeiten) bleibt der Player unverändert (gr.update())."""
    row, col = evt.index
    if col != REVIEW_PLAY_COL or row < 0 or row >= len(entries):
        return gr.update()
    clip_path = SEGMENTS_DIR / entries[row].clip_filename
    if not clip_path.exists():
        return gr.update()
    return str(clip_path)


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
            reg(snapshot_label, "label", "snapshot_label_label"); reg(snapshot_label, "placeholder", None)
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
            music_mode_radio = gr.Radio(
                [(t(DEFAULT_LANG, "music_mode_opt1"), "remove_music"),
                 (t(DEFAULT_LANG, "music_mode_opt2"), "cut_music_sections")],
                value="remove_music", label=t(DEFAULT_LANG, "music_mode_label"),
                info=t(DEFAULT_LANG, "music_mode_info"),
            )
            reg(music_mode_radio, "label", "music_mode_label"); reg(music_mode_radio, "info", None); reg(music_mode_radio, "music_choices", None)
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
                fn=handle_demucs_run, inputs=[denoise_checkbox, music_mode_radio],
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
                nisqa_threshold=NISQA_MOS_RED_THRESHOLD, asr_threshold=ASR_CONFIDENCE_RED_THRESHOLD,
            ))
            reg(tab4_desc_md, "tab4_desc", None)
            review_entries = gr.State([])

            with gr.Row():
                review_refresh_btn = gr.Button(t(DEFAULT_LANG, "review_refresh_btn"), size="sm")
                reg(review_refresh_btn, "value", "review_refresh_btn")
                nisqa_score_btn = gr.Button(t(DEFAULT_LANG, "nisqa_score_btn"), size="sm")
                reg(nisqa_score_btn, "value", "nisqa_score_btn")
                review_save_btn = gr.Button(t(DEFAULT_LANG, "review_save_btn"), variant="primary", size="sm")
                reg(review_save_btn, "value", "review_save_btn")

            review_status = gr.Markdown("")
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
            review_audio_player = gr.Audio(
                label=t(DEFAULT_LANG, "review_audio_label"), interactive=False, autoplay=True,
            )
            reg(review_audio_player, "label", "review_audio_label")

            review_refresh_btn.click(fn=handle_review_refresh, inputs=[], outputs=[review_entries, review_table])
            nisqa_score_btn.click(
                fn=handle_nisqa_score_all, inputs=[], outputs=[review_entries, review_table, review_status],
            )
            review_save_btn.click(
                fn=handle_review_save, inputs=[review_table, review_entries],
                outputs=[review_entries, review_table, review_status],
            )
            review_table.select(
                fn=handle_review_play, inputs=[review_entries], outputs=[review_audio_player],
            )

        with gr.TabItem(t(DEFAULT_LANG, "tab5_title")) as tab5:
            reg(tab5, "label", "tab5_title")
            tab5_desc_md = gr.Markdown(t(
                DEFAULT_LANG, "tab5_desc",
                xtts=EXPORT_SAMPLE_RATES["xtts"], rvc=EXPORT_SAMPLE_RATES["rvc"], lufs=TARGET_LUFS,
            ))
            reg(tab5_desc_md, "tab5_desc", None)
            export_btn = gr.Button(t(DEFAULT_LANG, "export_btn"), variant="primary")
            reg(export_btn, "value", "export_btn")
            export_log = gr.Textbox(label=t(DEFAULT_LANG, "log_label"), lines=8, interactive=False)
            reg(export_log, "label", "log_label")
            export_download = gr.File(label=t(DEFAULT_LANG, "export_download_label"))
            reg(export_download, "label", "export_download_label")
            export_btn.click(fn=handle_export_run, inputs=[], outputs=[export_log, export_download])

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
                updates.append(gr.update(placeholder=t(lang, "snapshot_label_placeholder")))
            elif kind == "info":
                updates.append(gr.update(info=t(lang, "music_mode_info")))
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
                    nisqa_threshold=NISQA_MOS_RED_THRESHOLD, asr_threshold=ASR_CONFIDENCE_RED_THRESHOLD,
                )))
            elif kind == "tab5_desc":
                updates.append(gr.update(value=t(
                    lang, "tab5_desc",
                    xtts=EXPORT_SAMPLE_RATES["xtts"], rvc=EXPORT_SAMPLE_RATES["rvc"], lufs=TARGET_LUFS,
                )))
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
