# Changelog

Versionsnummern rückwirkend rekonstruiert und nach folgender Regel neu
vergeben: **echte Erweiterungen** (neue Fähigkeiten) = Minor-Sprung
(0.X.0), **Bugfixes/kleine Korrekturen** = Patch-Sprung (0.X.Y). Ab hier
läuft die Versionierung über Git-Commits/-Tags weiter, nicht mehr nur
über diese Datei.

- **0.1.0** — Projekt-Grundgerüst, Tab 1 (Import: Datei-Upload, Mono/Resampling)
- **0.2.0** — Tab 2 (Music/Noise Removal via Demucs)
- **0.3.0** — Tab 3 (Silero-VAD-Segmentierung + whisperX-Transkription, Wortgrenzen-Snapping)
- **0.4.0** — Tab 4 (Review, Einzelclip-Ansicht mit Freigeben/Verwerfen)
- **0.5.0** — Tab 5 (Export XTTS/RVC), Umbenennung zu "Perfect Dataset GUI"
- **0.6.0** — Ordner-Import, htdemucs_ft, DeepFilterNet-Denoising, Fortschrittsbalken
- **0.7.0** — Fade-in/out + Silence-Trim an Clip-Grenzen
- **0.8.0** — Lautstärke-Normalisierung (LUFS) beim Export
- **0.9.0** — NISQA-Qualitäts-Scoring (eigener NISQA-Server, `start.sh`)
- **0.9.1** — Fix: Demucs-Subprozess nutzte falschen (System-)Python statt venv
- **0.9.2** — Fix: Demucs-CLI durch Python-API ersetzt (kaputte Lade-Routine im `demucs-torchcodec`-Fork)
- **0.9.3** — Fix: DeepFilterNet-Ein-/Ausgabe auf soundfile umgestellt (torchaudio-Bruch)
- **0.9.4** — Reset-Button, Stille-Erkennung als Sicherheitscheck, Silero-VAD-Lade-Fix
- **0.9.5** — Fix: whisperX-Audio-Laden umgangen (torchaudio-Bruch), Reset-beim-Beenden
- **0.10.0** — Snapshot-/Versionierungssystem eingeführt
- **0.10.1** — Fix: `LD_LIBRARY_PATH` für CUDA/cuDNN-Konflikt mit System-Installation
- **0.10.2** — Fix: `torchaudio.AudioMetaData`-Shim (pyannote-Importkette)
- **0.10.3** — Fix: zentrale `torchaudio`-Kompatibilitätsschicht (`list_audio_backends` u.a.)
- **0.11.0** — Review-Tab als Bulk-Edit-Tabelle, Reset-Bereich verkleinert, redundante Tab-3-Tabelle entfernt
- **0.12.0** — Musik-Umgang wählbar (entfernen vs. Abschnitte mit Musik komplett rausschneiden)
- **0.13.0** — Zweisprachige GUI (Deutsch/Englisch), Sprache der Eingangs-Audios wählbar, Button-Label gekürzt
- **0.13.1** — Fix-Versuch: `torch.load(weights_only=...)` (griff noch nicht, siehe 0.13.2)
- **0.13.2** — Fix: `weights_only` jetzt erzwungen statt nur vorbelegt (pytorch-lightning übergibt es explizit)
- **0.13.3** — Fix: `ctranslate2` auf cuDNN9-kompatible Version (>=4.5.0) gepinnt
- **0.13.4** — Fix: hartcodierter Sandbox-Pfad (`/home/claude/...`) beim Export-ZIP-Download entfernt (systemabhängiger Bug, wäre auf jedem anderen Rechner kaputt gewesen)
- **0.13.5** — README.md hinzugefügt (Setup-Anleitung, Voraussetzungen, Pipeline-Übersicht)
- **0.13.6** — Fix: Audio-Import scheiterte lautlos bei bestimmten WAV-Dateien (z.B. als .wav umbenannte MP3s) und meldete fälschlich 0,0s statt eines Fehlers; jetzt FFmpeg-Fallback (pydub) + sichtbare Fehlermeldung in der GUI. LICENSE (MIT) + Transparenz-Hinweis in README.md ergänzt (Projekt entstand im Dialog mit Claude)
- **0.13.7** — Fix: torch/torchaudio-Exaktpin (2.11.0) kollidierte mit whisperx-Versionen, die ctranslate2>=4.5.0 (cuDNN9-Fix) unterstützen — die brauchen torchaudio im 2.8.x-Bereich. Jetzt als Bereich statt Exaktversion gepinnt, damit uv eine gemeinsam kompatible Kombination findet
- **0.14.0** — DeepFilterNet in eigenen, separaten Denoise-Server ausgelagert (`denoise_server/`, eigene venv). Grund: unauflösbarer Konflikt zwischen deepfilternet (numpy<2.0) und modernen whisperx/ctranslate2-Versionen (numpy>=2.0) — beide können nicht in derselben Umgebung koexistieren. `start.sh` startet jetzt drei Prozesse (NISQA, Denoise, Haupt-GUI)
- **0.14.1** — Fix: torchcodec-Version auf torch-2.8.x-kompatiblen Bereich (0.6.x/0.7.x) eingeschränkt — lose Angabe hatte eine CUDA13-Version gezogen, die zu unserem CUDA12-torch nicht passte ("libnvrtc.so.13" fehlte)
- **0.14.2** — Fix: Denoise-Server-Kompat-Shim war unvollständig (fehlte der `torchaudio.backend`-Submodul-Importschutz, den DeepFilterNet tatsächlich braucht) — nachgeholt. `start.sh` prüft jetzt zusätzlich, ob `nisqalib` in der NISQA-venv wirklich installiert ist (nicht nur, ob die venv existiert), mit klarem Hinweis auf `nisqa_server/install.sh`
- **0.14.3** — start.sh: nisqalib-Check gibt jetzt eine ausführliche Diagnose aus (Python-Interpreter, site-packages-Pfad, gefundene nisqalib-Verzeichnisse, exakte Importfehlermeldung), statt nur "fehlt" zu melden — nisqalib wurde laut Installations-Log erfolgreich installiert, wird aber trotzdem nicht gefunden; Ursache noch ungeklärt
- **0.14.4** — Fix: nisqalib wird mit --no-deps installiert (wegen kaputter Versions-Pins), das übersprang auch dessen echte Laufzeit-Abhängigkeiten `tqdm` und `pyyaml` (NISQA-Original-Repo als Submodul) — jetzt explizit in nisqa_server/requirements.txt ergänzt
