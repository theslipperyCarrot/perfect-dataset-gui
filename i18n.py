"""
Übersetzungen für die Perfect Dataset GUI. Deutsch ist der Default.
Templates mit {platzhaltern} werden in app.py mit den aktuellen
config.py-Werten formatiert (Samplerate, Schwellwerte etc.), da die
sich nicht je Sprache unterscheiden, nur der umgebende Text.
"""

STRINGS = {
    "de": {
        "app_subtitle": "Der perfekte Audio-Datensatz für XTTS & RVC.",
        "workstandard_line": (
            "Interner Arbeitsstandard: **{sr} Hz**, Mono. "
            "Export-Ziele: XTTS ({xtts} Hz), RVC ({rvc} Hz)."
        ),
        "settings_title": "⚙️ Einstellungen",
        "language_label": "Sprache der Oberfläche",
        "input_language_label": "Sprache der Eingangs-Audios",
        "settings_snapshot_desc": (
            "**Snapshot**: sichert den aktuellen `data/`-Stand versioniert, unabhängig "
            "vom Reset (auch der automatische Reset beim Schließen der GUI rührt "
            "Snapshots nicht an). **Wiederherstellen** ersetzt den aktuellen Stand "
            "komplett durch den gewählten Snapshot."
        ),
        "snapshot_label_placeholder": "z.B. 'erste_10_kapitel'",
        "snapshot_label_label": "Label (optional)",
        "snapshot_create_btn": "📸 Snapshot erstellen",
        "snapshot_dropdown_label": "Vorhandene Snapshots",
        "snapshot_restore_btn": "⏪ Wiederherstellen",
        "snapshot_delete_btn": "🗑️ Löschen",
        "snapshot_log_label": "Snapshot-Log",
        "reset_confirm_label": "Ja, alle Zwischenergebnisse löschen",
        "reset_btn": "Zurücksetzen",
        "reset_log_label": "Reset-Log",
        "tab1_title": "1. Import",
        "tab1_desc": (
            "Lade die Rohaufnahmen hoch — einzeln/mehrfach auswählen oder "
            "gleich einen ganzen Ordner. Alle Dateien werden auf Mono + "
            "{sr} Hz gebracht und als WAV abgelegt."
        ),
        "file_input_label": "Audiodateien (Mehrfachauswahl)",
        "folder_input_label": "...oder ganzen Ordner wählen",
        "import_btn": "Importieren",
        "log_label": "Log",
        "raw_table_label": "Importierte Rohdateien",
        "file_col": "Datei",
        "tab2_title": "2. Music/Noise Removal",
        "tab2_desc": (
            "Entfernt Musik/Hintergrundgeräusche per Demucs (htdemucs_ft) "
            "aus allen importierten Rohdateien. "
            "Läuft auf GPU deutlich schneller. "
            "(Braucht FFmpeg als Systemabhängigkeit: `sudo apt install ffmpeg`, "
            "falls noch nicht vorhanden.)"
        ),
        "music_mode_label": "Umgang mit Musik",
        "music_mode_opt1": "Musik entfernen, Sprache behalten (Standard)",
        "music_mode_opt2": "Abschnitte mit Musik komplett herausschneiden",
        "music_mode_info": (
            "Erste Option: Demucs versucht Musik herauszurechnen, Sprache bleibt "
            "als (evtl. leicht unsauber) getrennte Spur erhalten. Zweite Option: "
            "Zeitabschnitte mit zu viel Musik-Restenergie werden komplett entfernt "
            "statt eine unsaubere Trennung zu behalten — höhere Datensatz-Qualität, "
            "aber ggf. weniger Material."
        ),
        "denoise_checkbox_label": "Zusätzliches Denoising (DeepFilterNet)",
        "demucs_btn": "Musik/Rauschen entfernen",
        "processed_table_label": "Bereinigte Dateien (data/processed)",
        "tab3_title": "3. Segmentierung + Transkription",
        "tab3_desc": (
            "Führt Silero-VAD-Segmentierung und whisperX-Transkription in einem "
            "Schritt aus. Clip-Grenzen werden automatisch auf Wortgrenzen "
            "eingerastet, damit keine Wörter abgeschnitten werden. "
            "Verarbeitet alle Dateien aus `data/processed`. Die erzeugten Clips "
            "siehst und bearbeitest du im **Review-Tab**."
        ),
        "device_choice_label": "Gerät für whisperX/VAD",
        "segment_btn": "Segmentieren & Transkribieren",
        "tab4_title": "4. Review",
        "tab4_desc": (
            "Alle Clips in einer Tabelle. Transkription direkt anklicken und "
            "bearbeiten. **NISQA bewerten** ruft den separaten NISQA-Server auf "
            "(MOS 1-5, rot bei < {nisqa_threshold}); die ASR-Konfidenz "
            "kommt automatisch aus whisperX (rot bei < {asr_threshold}, "
            "kann auf undeutliche Aussprache hindeuten). Häkchen bei **Löschen** setzen "
            "und **Änderungen speichern** klicken, um Clips endgültig zu entfernen "
            "(inkl. WAV-Datei) — alles, was danach noch in der Tabelle steht, landet "
            "im Export."
        ),
        "review_refresh_btn": "🔄 Neu laden",
        "nisqa_score_btn": "🎯 Alle Clips bewerten (NISQA)",
        "review_save_btn": "💾 Änderungen speichern",
        "review_col_nr": "#",
        "review_col_source": "Quelldatei",
        "review_col_transcript": "Transkription",
        "review_col_nisqa": "NISQA",
        "review_col_whisper": "Whisper",
        "review_col_delete": "Löschen",
        "tab5_title": "5. Export",
        "tab5_desc": (
            "Exportiert alle **freigegebenen** Clips parallel in beide Zielformate:\n"
            "- **XTTS**: LJSpeech-Struktur (`wavs/` + `metadata.csv`, {xtts} Hz)\n"
            "- **RVC**: reine WAV-Sammlung ({rvc} Hz), kein Text nötig\n\n"
            "Alle Clips werden dabei einheitlich auf **{lufs} LUFS** normalisiert "
            "(mit Peak-Limiting gegen Clipping), damit das Training nicht versehentlich "
            "Lautstärke-Schwankungen mitlernt.\n\n"
            "Clips ohne Text (z.B. nach dem Löschen des Transkripts im Review) "
            "werden nur für RVC exportiert, nicht für XTTS."
        ),
        "export_btn": "Export erstellen",
        "export_download_label": "Export als ZIP herunterladen",
    },
    "en": {
        "app_subtitle": "The perfect audio dataset for XTTS & RVC.",
        "workstandard_line": (
            "Internal working standard: **{sr} Hz**, mono. "
            "Export targets: XTTS ({xtts} Hz), RVC ({rvc} Hz)."
        ),
        "settings_title": "⚙️ Settings",
        "language_label": "Interface language",
        "input_language_label": "Input audio language",
        "settings_snapshot_desc": (
            "**Snapshot**: saves the current `data/` state as a version, independent "
            "of Reset (the automatic reset on closing the GUI doesn't touch "
            "snapshots). **Restore** completely replaces the current state "
            "with the selected snapshot."
        ),
        "snapshot_label_placeholder": "e.g. 'first_10_chapters'",
        "snapshot_label_label": "Label (optional)",
        "snapshot_create_btn": "📸 Create snapshot",
        "snapshot_dropdown_label": "Existing snapshots",
        "snapshot_restore_btn": "⏪ Restore",
        "snapshot_delete_btn": "🗑️ Delete",
        "snapshot_log_label": "Snapshot log",
        "reset_confirm_label": "Yes, delete all intermediate results",
        "reset_btn": "Reset",
        "reset_log_label": "Reset log",
        "tab1_title": "1. Import",
        "tab1_desc": (
            "Upload the raw recordings — select individually/multiple, or "
            "a whole folder at once. All files get converted to mono + "
            "{sr} Hz and stored as WAV."
        ),
        "file_input_label": "Audio files (multi-select)",
        "folder_input_label": "...or select a whole folder",
        "import_btn": "Import",
        "log_label": "Log",
        "raw_table_label": "Imported raw files",
        "file_col": "File",
        "tab2_title": "2. Music/Noise Removal",
        "tab2_desc": (
            "Removes music/background noise via Demucs (htdemucs_ft) "
            "from all imported raw files. "
            "Runs much faster on GPU. "
            "(Needs FFmpeg as a system dependency: `sudo apt install ffmpeg`, "
            "if not already installed.)"
        ),
        "music_mode_label": "How to handle music",
        "music_mode_opt1": "Remove music, keep speech (default)",
        "music_mode_opt2": "Cut out sections with music entirely",
        "music_mode_info": (
            "First option: Demucs tries to subtract the music, speech remains "
            "as a (possibly slightly imperfect) separated track. Second option: "
            "time sections with too much residual music energy get removed "
            "entirely instead of keeping an imperfect separation — higher "
            "dataset quality, but possibly less material."
        ),
        "denoise_checkbox_label": "Additional denoising (DeepFilterNet)",
        "demucs_btn": "Remove music/noise",
        "processed_table_label": "Cleaned files (data/processed)",
        "tab3_title": "3. Segmentation + Transcription",
        "tab3_desc": (
            "Runs Silero VAD segmentation and whisperX transcription in one "
            "step. Clip boundaries are automatically snapped to word "
            "boundaries so no words get cut off. "
            "Processes all files from `data/processed`. You can see and "
            "edit the resulting clips in the **Review tab**."
        ),
        "device_choice_label": "Device for whisperX/VAD",
        "segment_btn": "Segment & Transcribe",
        "tab4_title": "4. Review",
        "tab4_desc": (
            "All clips in one table. Click the transcript to edit it "
            "directly. **Score with NISQA** calls the separate NISQA server "
            "(MOS 1-5, red below {nisqa_threshold}); ASR confidence comes "
            "automatically from whisperX (red below {asr_threshold}, can "
            "indicate unclear pronunciation). Check **Delete** and click "
            "**Save changes** to permanently remove clips (incl. WAV file) "
            "— everything still in the table afterwards goes into the export."
        ),
        "review_refresh_btn": "🔄 Reload",
        "nisqa_score_btn": "🎯 Score all clips (NISQA)",
        "review_save_btn": "💾 Save changes",
        "review_col_nr": "#",
        "review_col_source": "Source file",
        "review_col_transcript": "Transcript",
        "review_col_nisqa": "NISQA",
        "review_col_whisper": "Whisper",
        "review_col_delete": "Delete",
        "tab5_title": "5. Export",
        "tab5_desc": (
            "Exports all **approved** clips in parallel into both target formats:\n"
            "- **XTTS**: LJSpeech structure (`wavs/` + `metadata.csv`, {xtts} Hz)\n"
            "- **RVC**: plain WAV collection ({rvc} Hz), no text needed\n\n"
            "All clips get uniformly normalized to **{lufs} LUFS** "
            "(with peak limiting against clipping), so training doesn't "
            "accidentally pick up on loudness variation.\n\n"
            "Clips without text (e.g. after deleting the transcript in Review) "
            "only get exported for RVC, not for XTTS."
        ),
        "export_btn": "Create export",
        "export_download_label": "Download export as ZIP",
    },
}


def t(lang: str, key: str, **kwargs) -> str:
    template = STRINGS.get(lang, STRINGS["de"]).get(key, key)
    return template.format(**kwargs) if kwargs else template
