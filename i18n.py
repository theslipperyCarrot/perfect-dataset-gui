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
            "Entfernt Musik/Hintergrundgeräusche aus allen importierten "
            "Rohdateien — wahlweise per Demucs oder per UVR (audio-separator). "
            "Läuft auf GPU deutlich schneller. "
            "(Braucht FFmpeg als Systemabhängigkeit: `sudo apt install ffmpeg`, "
            "falls noch nicht vorhanden.)"
        ),
        "engine_label": "Trenn-Engine",
        "engine_opt_demucs": "Demucs (htdemucs_ft, schnell)",
        "engine_opt_uvr": "UVR / audio-separator (BS-Roformer, oft sauberer)",
        "engine_info": (
            "Demucs ist schneller und läuft immer mit (Standard-Installation). "
            "UVR (Ultimate Vocal Remover, hier über das Paket audio-separator) "
            "gilt in der Community oft als sauberer bei der Vocal-Trennung, "
            "ist aber rechenintensiver und braucht eine eigene, separate "
            "Installation: bash separator_server/install.sh — ohne die läuft "
            "diese Option auf einen klaren Fehler statt eines stillen Fallbacks."
        ),
        "music_mode_label": "Umgang mit Musik",
        "music_mode_opt1": "Musik entfernen, Sprache behalten (Standard)",
        "music_mode_opt2": "Abschnitte mit Musik komplett herausschneiden",
        "music_mode_info": (
            "Erste Option: die Engine versucht Musik herauszurechnen, Sprache "
            "bleibt als (evtl. leicht unsauber) getrennte Spur erhalten. Zweite "
            "Option: Zeitabschnitte mit zu viel Musik-Restenergie werden "
            "komplett entfernt statt eine unsaubere Trennung zu behalten — "
            "höhere Datensatz-Qualität, aber ggf. weniger Material."
        ),
        "denoise_checkbox_label": "Zusätzliches Denoising (DeepFilterNet)",
        "demucs_btn": "Musik/Rauschen entfernen",
        "processed_table_label": "Bereinigte Dateien (data/processed)",
        "tab3_title": "3. Segmentierung + Transkription",
        "tab3_desc": (
            "Führt Silero-VAD-Segmentierung und whisperX-Transkription in einem "
            "Schritt aus. Clip-Grenzen werden automatisch auf Wortgrenzen "
            "eingerastet, damit keine Wörter abgeschnitten werden. "
            "Verarbeitet alle Dateien aus `data/processed`. Im Anschluss werden "
            "automatisch alle Clips per NISQA bewertet (kein separater Klick in "
            "Tab 4 mehr nötig). Die erzeugten Clips siehst und bearbeitest du im "
            "**Review-Tab**."
        ),
        "device_choice_label": "Gerät für whisperX/VAD",
        "segment_btn": "Segmentieren & Transkribieren",
        "tab4_title": "4. Review",
        "tab4_desc": (
            "Alle Clips in einer Tabelle. Transkription direkt anklicken und "
            "bearbeiten. Auf **▶** klicken lädt den Clip in den Player oben, "
            "zum Vergleich von Audio und Text. Der Player ist bearbeitbar: "
            "Trimm-Griffe an den Rändern der Wellenform ziehen, bestätigen, "
            "dann **✂️ Zuschnitt übernehmen** klicken, um die Clip-Grenzen "
            "direkt zu korrigieren. "
            "**NISQA bewerten** ruft den separaten NISQA-Server auf "
            "(MOS 1-5, rot bei < {nisqa_threshold}). **Round-Trip prüfen** "
            "transkribiert jeden Clip unabhängig neu und vergleicht das "
            "Ergebnis mit dem zugewiesenen Text (0-1, rot bei < {asr_threshold}) "
            "— deckt zuverlässiger als ein reiner Konfidenzwert auf, wenn ein "
            "Nachbarwort oder eine Pause mit ins Audio gerutscht ist. "
            "**Mangelhafte Clips automatisch neu schneiden** (erst NACH einem "
            "Round-Trip-Durchlauf sinnvoll) versucht für alle rot markierten "
            "Clips automatisch einen besseren Zuschnitt aus der Quelldatei — "
            "bringt das keine Verbesserung, bleibt der Clip unverändert und "
            "weiterhin markiert, für die manuelle Kontrolle per Trimm-Tool. "
            "Häkchen bei **Löschen** setzen und **Änderungen speichern** "
            "klicken, um Clips endgültig zu entfernen (inkl. WAV-Datei) — "
            "alles, was danach noch in der Tabelle steht, landet im Export."
        ),
        "review_refresh_btn": "🔄 Neu laden",
        "nisqa_score_btn": "🎯 Alle Clips neu bewerten (NISQA)",
        "roundtrip_btn": "🔁 Round-Trip prüfen",
        "reclip_btn": "🔧 Mangelhafte Clips automatisch neu schneiden",
        "review_save_btn": "💾 Änderungen speichern",
        "trim_apply_btn": "✂️ Zuschnitt übernehmen",
        "trim_status_none": "Kein Clip ausgewählt oder keine Änderung im Player.",
        "trim_status_saved": "'{filename}' aktualisiert (neue Länge: {duration}s).",
        "review_col_nr": "#",
        "review_col_source": "Quelldatei",
        "review_col_transcript": "Transkription",
        "review_col_play": "▶",
        "review_audio_label": "Ausgewählter Clip",
        "review_col_nisqa": "NISQA",
        "review_col_whisper": "Round-Trip",
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

        "dataset_summary_title": "📊 Dataset-Überblick",
        "dataset_summary_refresh_btn": "🔄 Überblick aktualisieren",
        "dataset_summary_empty": "Noch keine Clips vorhanden — erst Tab 3 (und optional Tab 4) durchlaufen.",
        "dataset_summary_text": (
            "**{count} Clips**, Gesamtdauer **{hms}** (h:mm:ss). "
            "Clip-Länge: Ø {avg}s (min {min}s, max {max}s). NISQA-Ø: {nisqa}."
        ),
        "dataset_summary_nisqa_none": "noch nicht bewertet",
        "dataset_summary_chart_x": "Clip-Länge",
        "dataset_summary_chart_y": "Anzahl",
        "dataset_summary_chart_label": "Verteilung der Clip-Längen",
        "speaker_name_label": "Sprecher-Name (optional)",
        "speaker_name_placeholder": "z.B. max-mustermann — wird für den Export-Dateinamen verwendet",

        "tab_settings_title": "⚙️ Einstellungen",
        "settings_desc": (
            "Workflow-Parameter, die das Ergebnis beeinflussen — hier direkt "
            "anpassbar, ohne Code anzufassen. Wirkt sofort beim nächsten "
            "Durchlauf des jeweiligen Tabs, kein Neustart nötig. Rein "
            "Technisches (Server-Ports, Pfade, Samplerate) bleibt bewusst "
            "in `config.py`."
        ),
        "settings_group_segmentation": "Segmentierung (Tab 3)",
        "settings_group_music": "Music/Noise Removal (Tab 2)",
        "settings_group_quality": "Qualitäts-Schwellwerte (Tab 4)",
        "settings_group_export": "Export (Tab 5)",
        "settings_save_btn": "💾 Speichern",
        "settings_reset_btn": "↺ Auf Werkseinstellungen zurücksetzen",
        "settings_status_saved": "Gespeichert — wirkt ab dem nächsten Durchlauf.",
        "settings_status_reset": "Auf Werkseinstellungen zurückgesetzt.",

        "setting_MIN_CLIP_DURATION_S_label": "Mindest-Clip-Länge (s)",
        "setting_MIN_CLIP_DURATION_S_info": "Kürzere Abschnitte werden mit dem vorherigen Clip zusammengelegt oder verworfen.",
        "setting_MAX_CLIP_DURATION_S_label": "Maximale Clip-Länge (s)",
        "setting_MAX_CLIP_DURATION_S_info": "Clips werden spätestens hier hart getrennt (nie mitten im Wort).",
        "setting_MIN_SILENCE_GAP_S_label": "Mindest-Sprechpause (s)",
        "setting_MIN_SILENCE_GAP_S_info": "Ab dieser Pausenlänge zwischen zwei Wörtern wird zusätzlich zu Satzenden/Kommas getrennt.",
        "setting_SILERO_VAD_THRESHOLD_label": "VAD-Sprach-Schwelle",
        "setting_SILERO_VAD_THRESHOLD_info": "Ab welcher Sprach-Wahrscheinlichkeit (0-1) ein Abschnitt als Sprache gilt.",
        "setting_WORD_BOUNDARY_PAD_S_label": "Wortrand-Polster (s)",
        "setting_WORD_BOUNDARY_PAD_S_info": "Sicherheitsabstand vor dem ersten/nach dem letzten Wort eines Clips, gegen abgeschnittene Wortanfänge/-enden.",
        "setting_SILENCE_TRIM_TOP_DB_label": "Rand-Trim-Schwelle (dB)",
        "setting_SILENCE_TRIM_TOP_DB_info": "Wie viel leiser als der lauteste Punkt im Clip als 'Stille' gilt und an den Rändern abgeschnitten wird.",
        "setting_SILENCE_TRIM_MAX_S_label": "Max. Rand-Trim (s)",
        "setting_SILENCE_TRIM_MAX_S_info": "Obergrenze, wie viel der Rand-Trim pro Seite maximal wegschneiden darf (Schutz gegen verschluckte Wörter).",
        "setting_FADE_DURATION_S_label": "Fade-in/-out (s)",
        "setting_FADE_DURATION_S_info": "Kurzes Ein-/Ausblenden an jedem Clip-Rand gegen Klick-/Atem-Reste.",
        "setting_WHISPERX_BATCH_SIZE_label": "whisperX-Batch-Größe",
        "setting_WHISPERX_BATCH_SIZE_info": "Wie viele Audio-Chunks whisperX gleichzeitig verarbeitet. Höher = schneller, aber mehr VRAM-Bedarf. Bei 'CUDA out of memory' hier verkleinern.",
        "setting_RECLIP_MARGIN_S_label": "Neu-Zuschnitt-Suchfenster (s)",
        "setting_RECLIP_MARGIN_S_info": "Wie viel zusätzlicher Kontext beim automatischen Neu-Zuschnitt links/rechts aus der Quelldatei geladen wird (erster Versuch; jeder weitere Versuch verdoppelt/verdreifacht diesen Wert).",
        "setting_RECLIP_MAX_ATTEMPTS_label": "Neu-Zuschnitt-Versuche",
        "setting_RECLIP_MAX_ATTEMPTS_info": "Wie oft mit größer werdendem Suchfenster neu versucht wird, bevor beim automatischen Neu-Zuschnitt aufgegeben wird.",
        "setting_MUSIC_CUT_WINDOW_S_label": "Musik-Analysefenster (s)",
        "setting_MUSIC_CUT_WINDOW_S_info": "Fenstergröße für die Musik-Energie-Analyse (Modus 'Musik-Abschnitte rausschneiden').",
        "setting_MUSIC_CUT_ENERGY_RATIO_label": "Musik-Schwellwert",
        "setting_MUSIC_CUT_ENERGY_RATIO_info": "Anteil Musik-Energie an der Gesamtenergie, ab dem ein Zeitfenster als 'zu musikalisch' gilt und entfernt wird.",
        "setting_SILENCE_SKIP_WINDOW_S_label": "Stille-Analysefenster (s)",
        "setting_SILENCE_SKIP_WINDOW_S_info": "Fenstergröße für die Stille-Vorprüfung vor der Trennung (Tab 2).",
        "setting_SILENCE_SKIP_MIN_DURATION_S_label": "Mindest-Stille zum Überspringen (s)",
        "setting_SILENCE_SKIP_MIN_DURATION_S_info": "Ab dieser Länge wird eine Stille-Passage nicht mehr durch die Trennung geschickt, sondern unverändert übernommen.",
        "setting_SILENCE_SKIP_RMS_THRESHOLD_label": "Stille-Pegel-Schwelle",
        "setting_SILENCE_SKIP_RMS_THRESHOLD_info": "Lautstärke-Schwelle (RMS), unterhalb der ein Zeitfenster als Stille gilt.",
        "setting_SILENCE_SKIP_CROSSFADE_S_label": "Crossfade an Stille-Nahtstellen (s)",
        "setting_SILENCE_SKIP_CROSSFADE_S_info": "Überblendung zwischen übersprungener Stille und Trennungs-Ergebnis, gegen hörbare Klicks.",
        "setting_NISQA_MOS_RED_THRESHOLD_label": "NISQA-Warnschwelle",
        "setting_NISQA_MOS_RED_THRESHOLD_info": "NISQA-Score (1-5), unterhalb dessen ein Clip im Review-Tab rot markiert wird.",
        "setting_ASR_CONFIDENCE_RED_THRESHOLD_label": "Round-Trip-Warnschwelle",
        "setting_ASR_CONFIDENCE_RED_THRESHOLD_info": "Round-Trip-Score (0-1, 1=perfekte Übereinstimmung mit einer unabhängigen Neu-Transkription), unterhalb dessen ein Clip im Review-Tab rot markiert wird.",
        "setting_TARGET_LUFS_label": "Ziel-Lautstärke (LUFS)",
        "setting_TARGET_LUFS_info": "Einheitliche Lautstärke-Normalisierung beim Export (XTTS + RVC).",
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
            "Removes music/background noise from all imported raw files — "
            "either via Demucs or via UVR (audio-separator). "
            "Runs much faster on GPU. "
            "(Needs FFmpeg as a system dependency: `sudo apt install ffmpeg`, "
            "if not already installed.)"
        ),
        "engine_label": "Separation engine",
        "engine_opt_demucs": "Demucs (htdemucs_ft, fast)",
        "engine_opt_uvr": "UVR / audio-separator (BS-Roformer, often cleaner)",
        "engine_info": (
            "Demucs is faster and always available (default install). UVR "
            "(Ultimate Vocal Remover, here via the audio-separator package) is "
            "often considered cleaner for vocal separation in the community, "
            "but is more compute-intensive and needs its own separate install: "
            "bash separator_server/install.sh — without it, this option fails "
            "clearly instead of silently falling back."
        ),
        "music_mode_label": "How to handle music",
        "music_mode_opt1": "Remove music, keep speech (default)",
        "music_mode_opt2": "Cut out sections with music entirely",
        "music_mode_info": (
            "First option: the engine tries to subtract the music, speech "
            "remains as a (possibly slightly imperfect) separated track. "
            "Second option: time sections with too much residual music energy "
            "get removed entirely instead of keeping an imperfect separation "
            "— higher dataset quality, but possibly less material."
        ),
        "denoise_checkbox_label": "Additional denoising (DeepFilterNet)",
        "demucs_btn": "Remove music/noise",
        "processed_table_label": "Cleaned files (data/processed)",
        "tab3_title": "3. Segmentation + Transcription",
        "tab3_desc": (
            "Runs Silero VAD segmentation and whisperX transcription in one "
            "step. Clip boundaries are automatically snapped to word "
            "boundaries so no words get cut off. "
            "Processes all files from `data/processed`. Afterwards all clips "
            "are automatically scored with NISQA (no separate click in Tab 4 "
            "needed anymore). You can see and edit the resulting clips in the "
            "**Review tab**."
        ),
        "device_choice_label": "Device for whisperX/VAD",
        "segment_btn": "Segment & Transcribe",
        "tab4_title": "4. Review",
        "tab4_desc": (
            "All clips in one table. Click the transcript to edit it "
            "directly. Click **▶** to load the clip into the player above, "
            "to compare audio and text. The player is editable: drag the "
            "trim handles at the edges of the waveform, confirm, then click "
            "**✂️ Apply trim** to correct the clip boundaries directly. "
            "**Score with NISQA** calls the separate NISQA server "
            "(MOS 1-5, red below {nisqa_threshold}). **Round-trip check** "
            "re-transcribes each clip independently and compares the result "
            "to the assigned text (0-1, red below {asr_threshold}) — more "
            "reliably catches a neighboring word or pause that slipped into "
            "the audio than a plain confidence value. "
            "**Auto-recut deficient clips** (best used AFTER a round-trip "
            "run) tries to find a better cut from the source file for every "
            "clip flagged red — if that brings no improvement, the clip "
            "stays unchanged and still flagged, for manual review with the "
            "trim tool. Check **Delete** and click **Save changes** to "
            "permanently remove clips (incl. WAV file) — everything still "
            "in the table afterwards goes into the export."
        ),
        "review_refresh_btn": "🔄 Reload",
        "nisqa_score_btn": "🎯 Re-score all clips (NISQA)",
        "roundtrip_btn": "🔁 Round-trip check",
        "reclip_btn": "🔧 Auto-recut deficient clips",
        "review_save_btn": "💾 Save changes",
        "trim_apply_btn": "✂️ Apply trim",
        "trim_status_none": "No clip selected, or no change in the player.",
        "trim_status_saved": "'{filename}' updated (new length: {duration}s).",
        "review_col_nr": "#",
        "review_col_source": "Source file",
        "review_col_transcript": "Transcript",
        "review_col_play": "▶",
        "review_audio_label": "Selected clip",
        "review_col_nisqa": "NISQA",
        "review_col_whisper": "Round-Trip",
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

        "dataset_summary_title": "📊 Dataset overview",
        "dataset_summary_refresh_btn": "🔄 Refresh overview",
        "dataset_summary_empty": "No clips yet — run Tab 3 (and optionally Tab 4) first.",
        "dataset_summary_text": (
            "**{count} clips**, total duration **{hms}** (h:mm:ss). "
            "Clip length: avg {avg}s (min {min}s, max {max}s). Avg NISQA: {nisqa}."
        ),
        "dataset_summary_nisqa_none": "not scored yet",
        "dataset_summary_chart_x": "Clip length",
        "dataset_summary_chart_y": "Count",
        "dataset_summary_chart_label": "Clip length distribution",
        "speaker_name_label": "Speaker name (optional)",
        "speaker_name_placeholder": "e.g. jane-doe — used for the export filename",

        "tab_settings_title": "⚙️ Settings",
        "settings_desc": (
            "Workflow parameters that affect the result — adjustable right "
            "here, no need to touch code. Takes effect on the next run of "
            "the relevant tab, no restart needed. Purely technical settings "
            "(server ports, paths, sample rate) deliberately stay in "
            "`config.py`."
        ),
        "settings_group_segmentation": "Segmentation (Tab 3)",
        "settings_group_music": "Music/Noise Removal (Tab 2)",
        "settings_group_quality": "Quality thresholds (Tab 4)",
        "settings_group_export": "Export (Tab 5)",
        "settings_save_btn": "💾 Save",
        "settings_reset_btn": "↺ Reset to defaults",
        "settings_status_saved": "Saved — takes effect on the next run.",
        "settings_status_reset": "Reset to defaults.",

        "setting_MIN_CLIP_DURATION_S_label": "Minimum clip length (s)",
        "setting_MIN_CLIP_DURATION_S_info": "Shorter sections get merged into the previous clip or dropped.",
        "setting_MAX_CLIP_DURATION_S_label": "Maximum clip length (s)",
        "setting_MAX_CLIP_DURATION_S_info": "Clips are hard-cut here at the latest (never mid-word).",
        "setting_MIN_SILENCE_GAP_S_label": "Minimum speech pause (s)",
        "setting_MIN_SILENCE_GAP_S_info": "A pause between two words at least this long is used as a cut point, in addition to sentence ends/commas.",
        "setting_SILERO_VAD_THRESHOLD_label": "VAD speech threshold",
        "setting_SILERO_VAD_THRESHOLD_info": "Speech probability (0-1) above which a section counts as speech.",
        "setting_WORD_BOUNDARY_PAD_S_label": "Word boundary padding (s)",
        "setting_WORD_BOUNDARY_PAD_S_info": "Safety margin before the first/after the last word of a clip, against clipped word starts/ends.",
        "setting_SILENCE_TRIM_TOP_DB_label": "Edge trim threshold (dB)",
        "setting_SILENCE_TRIM_TOP_DB_info": "How much quieter than the loudest point in the clip counts as 'silence' and gets trimmed off the edges.",
        "setting_SILENCE_TRIM_MAX_S_label": "Max. edge trim (s)",
        "setting_SILENCE_TRIM_MAX_S_info": "Upper limit on how much the edge trim can cut per side (protects against swallowed words).",
        "setting_FADE_DURATION_S_label": "Fade-in/-out (s)",
        "setting_FADE_DURATION_S_info": "Short fade at each clip edge against click/breath residue.",
        "setting_WHISPERX_BATCH_SIZE_label": "whisperX batch size",
        "setting_WHISPERX_BATCH_SIZE_info": "How many audio chunks whisperX processes at once. Higher = faster, but needs more VRAM. Lower this on 'CUDA out of memory'.",
        "setting_RECLIP_MARGIN_S_label": "Auto-recut search window (s)",
        "setting_RECLIP_MARGIN_S_info": "How much extra context gets loaded left/right from the source file for the automatic re-cut (first attempt; each further attempt doubles/triples this value).",
        "setting_RECLIP_MAX_ATTEMPTS_label": "Auto-recut attempts",
        "setting_RECLIP_MAX_ATTEMPTS_info": "How many times to retry with a growing search window before giving up on the automatic re-cut.",
        "setting_MUSIC_CUT_WINDOW_S_label": "Music analysis window (s)",
        "setting_MUSIC_CUT_WINDOW_S_info": "Window size for the music-energy analysis ('cut out music sections' mode).",
        "setting_MUSIC_CUT_ENERGY_RATIO_label": "Music threshold",
        "setting_MUSIC_CUT_ENERGY_RATIO_info": "Share of music energy in total energy above which a time window counts as 'too musical' and gets removed.",
        "setting_SILENCE_SKIP_WINDOW_S_label": "Silence analysis window (s)",
        "setting_SILENCE_SKIP_WINDOW_S_info": "Window size for the silence pre-check before separation (Tab 2).",
        "setting_SILENCE_SKIP_MIN_DURATION_S_label": "Minimum silence to skip (s)",
        "setting_SILENCE_SKIP_MIN_DURATION_S_info": "A silent passage at least this long is no longer sent through separation, but kept as-is.",
        "setting_SILENCE_SKIP_RMS_THRESHOLD_label": "Silence level threshold",
        "setting_SILENCE_SKIP_RMS_THRESHOLD_info": "Loudness threshold (RMS) below which a time window counts as silence.",
        "setting_SILENCE_SKIP_CROSSFADE_S_label": "Crossfade at silence seams (s)",
        "setting_SILENCE_SKIP_CROSSFADE_S_info": "Blend between skipped silence and separation result, against audible clicks.",
        "setting_NISQA_MOS_RED_THRESHOLD_label": "NISQA warning threshold",
        "setting_NISQA_MOS_RED_THRESHOLD_info": "NISQA score (1-5) below which a clip is flagged red in the Review tab.",
        "setting_ASR_CONFIDENCE_RED_THRESHOLD_label": "Round-trip warning threshold",
        "setting_ASR_CONFIDENCE_RED_THRESHOLD_info": "Round-trip score (0-1, 1=perfect match with an independent re-transcription) below which a clip is flagged red in the Review tab.",
        "setting_TARGET_LUFS_label": "Target loudness (LUFS)",
        "setting_TARGET_LUFS_info": "Uniform loudness normalization on export (XTTS + RVC).",
    },
}


def t(lang: str, key: str, **kwargs) -> str:
    template = STRINGS.get(lang, STRINGS["de"]).get(key, key)
    return template.format(**kwargs) if kwargs else template
