#!/bin/bash
# Installiert den Vocal-Separation-Server (audio-separator/UVR-Modelle) in
# einer eigenen venv — isoliert vom Hauptprojekt (siehe requirements.txt).
set -e
cd "$(dirname "$0")"

echo "Lege venv an..."
uv venv

echo "Installiere Separator-Server-Abhängigkeiten..."
uv pip install -r requirements.txt

echo "Separator-Server-Installation abgeschlossen."
echo "Hinweis: Das Trenn-Modell wird beim ERSTEN Aufruf automatisch"
echo "heruntergeladen (mehrere hundert MB) -> dafür wird beim ersten"
echo "Durchlauf in Tab 2 (mit UVR als Engine) Internet gebraucht."
