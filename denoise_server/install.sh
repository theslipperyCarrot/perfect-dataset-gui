#!/bin/bash
# Installiert den Denoise-Server (DeepFilterNet) in einer eigenen venv —
# isoliert vom Hauptprojekt wegen des numpy<2.0-Konflikts mit whisperx/
# ctranslate2 (siehe requirements.txt-Kommentar).
set -e
cd "$(dirname "$0")"

echo "Lege venv an..."
uv venv

echo "Installiere Denoise-Server-Abhängigkeiten..."
uv pip install -r requirements.txt

echo "Denoise-Server-Installation abgeschlossen."
