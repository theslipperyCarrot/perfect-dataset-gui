# 🎙️ Perfect Dataset GUI

A local Gradio tool that turns raw recordings (e.g. audiobook/podcast material)
into a clean, training-ready audio dataset for **XTTS** fine-tuning and
**RVC** training — from raw file to finished, exported dataset in one
continuous pipeline.

Bilingual interface (German/English), German as default.

## About this project

This tool was built entirely in dialogue with **Claude** (Anthropic) — from
the initial project structure to individual bugfixes. I deliberately bring
little domain expertise in audio processing/ML myself; the technical
decisions, the code, and the debugging work come practically entirely from
Claude, while I tested, gave feedback, and set the direction. That also
means there may be places an experienced developer would solve
differently/better — hints, pull requests, and criticism are explicitly
welcome.

## Screenshots

## Pipeline

1. **Import** — Upload audio files (single, multiple, or a whole folder),
   automatic conversion to mono + a consistent sample rate.
2. **Music/Noise Removal** — Separate out music/background noise via Demucs
   (`htdemucs_ft`), with optional additional DeepFilterNet denoising.
   Selectable: separate out music *or* discard sections containing music
   entirely (higher dataset quality when separation is imperfect).
3. **Segmentation + Transcription** — Silero VAD detects speech segments,
   whisperX transcribes with word timestamps; clip boundaries snap to word
   boundaries, with fade-in/-out and silence trim at the edges.
4. **Review** — All clips in one editable table: correct the transcript
   directly, optional NISQA quality score + whisperX confidence (both
   highlighted red for poor values), delete individual clips.
5. **Export** — In parallel as an LJSpeech structure (XTTS) and a plain WAV
   collection (RVC), uniformly loudness-normalized (LUFS).

Also included: a snapshot/restore system for the data state, a reset
function, and automatic reset on exit.

## Requirements

- **Linux** (tested on Linux Mint/Zorin; other distros should work, possibly
  with a different package manager for FFmpeg)
- **NVIDIA GPU + current driver** strongly recommended (CUDA acceleration
  for Demucs/whisperX/DeepFilterNet/NISQA). A CPU option exists in the
  segmentation tab but is not currently fully tested/optimized.
- **[uv](https://docs.astral.sh/uv/)** as the Python package manager
- **FFmpeg**: `sudo apt install ffmpeg` (Debian/Ubuntu-based; use
  `dnf`/`pacman`/... accordingly on other distros)
- **Internet on first start** — Demucs, whisperX, Silero VAD, DeepFilterNet,
  and NISQA model weights are downloaded on the first run (several GB) and
  cached locally afterward.

## Installation

Three separate Python environments: the main GUI, the NISQA quality server,
and the denoise server (DeepFilterNet) — both as their own independent
venvs, since they have dependencies incompatible with the main project (see
the `requirements.txt` comments).

```
# Main GUI
uv venv
uv pip install -r requirements.txt

# NISQA server (separate, own venv)
bash nisqa_server/install.sh

# Denoise server (separate, own venv)
bash denoise_server/install.sh
```

## Starting

```
bash start.sh
```

Starts all servers (NISQA + Denoise in the background, then the main GUI)
and opens the GUI at `http://127.0.0.1:7860`. On error, the terminal window
stays open (even when started via double-click) so the message stays
readable.

To stop: `Ctrl+C` in the terminal — this automatically clears all
intermediate results (`data/`) for a clean restart. Save important
intermediate states beforehand via the **Snapshot** button in the settings
if you want to keep them.

## Configuration

Central settings (sample rates, thresholds, model selection, target
directories) in `config.py` — explained there with comments for what each
value does.

## Known Limitations

- Only tested as practical with an NVIDIA GPU; a CPU mode exists, but
  `WHISPERX_COMPUTE_TYPE` is hardcoded to `float16`, which doesn't work well
  on CPU (would need to be switched to `int8`).
- `torch`/`torchaudio` are pinned to a specific version (see
  `requirements.txt`), since several dependencies (Demucs fork,
  DeepFilterNet, whisperX/pyannote) aren't consistently compatible with the
  latest `torchaudio` APIs. Compatibility layer for this:
  `modules/torchaudio_compat.py`.
- NISQA server runs on port 8050, denoise server on port 8051 (both
  changeable in `config.py`).

Details on version history: [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE). Free to use, modify, and redistribute, including
commercially, with attribution.

---

# 🎙️ Perfect Dataset GUI (Deutsch)

Ein lokales Gradio-Tool, das aus Rohaufnahmen (z.B. Hörbuch-/Podcast-Material)
einen sauberen, trainingsfertigen Audio-Datensatz für **XTTS**-Finetuning und
**RVC**-Training macht — von der Rohdatei bis zum fertigen, exportierten
Datensatz in einer durchgehenden Pipeline.

Oberfläche zweisprachig (Deutsch/Englisch), Deutsch als Standard.

## Über dieses Projekt

Dieses Tool ist komplett im Dialog mit **Claude** (Anthropic) entstanden — von
der ersten Projektstruktur bis zu den einzelnen Bugfixes. Ich selbst bringe
dabei bewusst wenig Fachwissen in Audio-Verarbeitung/ML mit; die technischen
Entscheidungen, der Code und die Debugging-Arbeit stammen praktisch komplett
von Claude, ich habe getestet, Rückmeldung gegeben und die Richtung
vorgegeben. Das heißt auch: Es kann Stellen geben, die ein erfahrener
Entwickler anders/besser lösen würde — Hinweise, Pull Requests und Kritik
sind ausdrücklich willkommen.

## Screenshots

## Pipeline

1. **Import** — Audiodateien (einzeln, mehrfach oder ganzer Ordner) hochladen,
   automatische Konvertierung auf Mono + einheitliche Samplerate.
2. **Music/Noise Removal** — Musik/Hintergrundgeräusche per Demucs
   (`htdemucs_ft`) heraustrennen, optional zusätzliches DeepFilterNet-
   Denoising. Wählbar: Musik heraustrennen *oder* Abschnitte mit Musik
   komplett verwerfen (höhere Datensatz-Qualität bei unsauberer Trennung).
3. **Segmentierung + Transkription** — Silero VAD erkennt Sprachabschnitte,
   whisperX transkribiert mit Wort-Timestamps; Clip-Grenzen werden auf
   Wortgrenzen eingerastet, Fade-in/-out und Silence-Trim an den Rändern.
4. **Review** — alle Clips in einer bearbeitbaren Tabelle: Transkript direkt
   korrigieren, optional NISQA-Qualitätsbewertung + whisperX-Konfidenz
   (beide rot markiert bei schlechten Werten), einzelne Clips löschen.
5. **Export** — parallel als LJSpeech-Struktur (XTTS) und reine WAV-Sammlung
   (RVC), einheitlich lautstärke-normalisiert (LUFS).

Dazu: Snapshot-/Restore-System für den Datenstand, Reset-Funktion,
automatischer Reset beim Beenden.

## Voraussetzungen

- **Linux** (getestet auf Linux Mint/Zorin; andere Distros sollten
  funktionieren, ggf. anderer Paketmanager für FFmpeg)
- **NVIDIA-GPU + aktueller Treiber** dringend empfohlen (CUDA-Beschleunigung
  für Demucs/whisperX/DeepFilterNet/NISQA). Eine CPU-Option existiert im
  Segmentierungs-Tab, ist aber aktuell nicht durchgetestet/optimiert.
- **[uv](https://docs.astral.sh/uv/)** als Python-Paketmanager
- **FFmpeg**: `sudo apt install ffmpeg` (Debian/Ubuntu-basiert; bei anderen
  Distros entsprechend `dnf`/`pacman`/...)
- **Internet beim ersten Start** — Demucs-, whisperX-, Silero-VAD-,
  DeepFilterNet- und NISQA-Modellgewichte werden beim ersten Lauf
  heruntergeladen (mehrere GB) und danach lokal gecacht.

## Installation

Drei getrennte Python-Umgebungen: die Haupt-GUI, der NISQA-Qualitäts-Server
und der Denoise-Server (DeepFilterNet) — beide als eigene, unabhängige venvs,
da sie mit dem Hauptprojekt inkompatible Abhängigkeiten haben (siehe
`requirements.txt`-Kommentare).

```
# Haupt-GUI
uv venv
uv pip install -r requirements.txt

# NISQA-Server (separat, eigene venv)
bash nisqa_server/install.sh

# Denoise-Server (separat, eigene venv)
bash denoise_server/install.sh
```

## Starten

```
bash start.sh
```

Startet alle Server (NISQA + Denoise im Hintergrund, dann die Haupt-GUI) und
öffnet die GUI unter `http://127.0.0.1:7860`. Bei einem Fehler bleibt das
Terminal-Fenster offen (auch bei Start per Doppelklick), damit die Meldung
lesbar ist.

Zum Beenden: `Strg+C` im Terminal — das leert automatisch alle
Zwischenergebnisse (`data/`) für einen sauberen Neustart. Wichtige
Zwischenstände vorher über den **Snapshot**-Button in den Einstellungen
sichern, falls sie erhalten bleiben sollen.

## Konfiguration

Zentrale Einstellungen (Samplerates, Schwellwerte, Modell-Auswahl,
Zielverzeichnisse) in `config.py` — dort mit Kommentar erklärt, wofür jeder
Wert steht.

## Bekannte Einschränkungen

- Nur mit NVIDIA-GPU praxistauglich getestet; CPU-Modus vorhanden, aber
  `WHISPERX_COMPUTE_TYPE` ist fest auf `float16` gestellt, was auf CPU nicht
  gut funktioniert (müsste auf `int8` umgestellt werden).
- `torch`/`torchaudio` sind auf eine bestimmte Version gepinnt (siehe
  `requirements.txt`), da mehrere Abhängigkeiten (Demucs-Fork, DeepFilterNet,
  whisperX/pyannote) nicht durchgängig mit den neuesten `torchaudio`-APIs
  kompatibel sind. Kompatibilitätsschicht dafür: `modules/torchaudio_compat.py`.
- NISQA-Server läuft auf Port 8050, Denoise-Server auf Port 8051 (beide in
  `config.py` änderbar).

Details zur Versionshistorie: [CHANGELOG.md](CHANGELOG.md).

## Lizenz

MIT — siehe [LICENSE](LICENSE). Freie Nutzung, Veränderung und Weitergabe,
auch kommerziell, mit Namensnennung.
