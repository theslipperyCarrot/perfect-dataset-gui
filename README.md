# 🎙️ Perfect Dataset GUI

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

<!-- TODO: Screenshots der 5 Tabs hier einfügen -->
<!-- ![Tab 1: Import](docs/screenshot-tab1.png) -->
<!-- ![Tab 3: Segmentierung](docs/screenshot-tab3.png) -->
<!-- ![Tab 4: Review](docs/screenshot-tab4.png) -->

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

Zwei getrennte Python-Umgebungen: die Haupt-GUI und der NISQA-Qualitäts-
Server (eigene, unabhängige venv, da eigene Torch-/Abhängigkeits-Version).

```bash
# Haupt-GUI
uv venv
uv pip install -r requirements.txt

# NISQA-Server (separat, eigene venv)
bash nisqa_server/install.sh
```

## Starten

```bash
bash start.sh
```

Startet beide Server (NISQA im Hintergrund + Haupt-GUI) und öffnet die GUI
unter `http://127.0.0.1:7860`. Bei einem Fehler bleibt das Terminal-Fenster
offen (auch bei Start per Doppelklick), damit die Meldung lesbar ist.

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
- NISQA-Server läuft aktuell fest auf Port 8050 (in `config.py` änderbar).

Details zur Versionshistorie: [CHANGELOG.md](CHANGELOG.md).

## Lizenz

MIT — siehe [LICENSE](LICENSE). Freie Nutzung, Veränderung und Weitergabe,
auch kommerziell, mit Namensnennung.
