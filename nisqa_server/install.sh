#!/bin/bash
# Installiert den NISQA-Server in einer eigenen venv.
#
# nisqalib (1.1.0) deklariert veraltete, feste Versionsangaben
# (numpy==1.20.3, pandas==1.3.5), die mit aktuellem Python gar nicht
# installierbar sind. Die Bibliothek braucht diese exakten Versionen
# zur Laufzeit nicht wirklich — deshalb installieren wir zuerst
# kompatible, aktuelle Versionen der echten Abhängigkeiten und ziehen
# nisqalib danach mit --no-deps nach (überspringt dessen kaputte Pins).
set -e
cd "$(dirname "$0")"

echo "Lege venv an..."
uv venv

echo "Installiere Basis-Abhängigkeiten..."
uv pip install -r requirements.txt

echo "Installiere nisqalib (ohne dessen veraltete Versions-Pins)..."
uv pip install --no-deps "nisqalib @ git+https://github.com/kale4eat/nisqalib"

echo "NISQA-Server-Installation abgeschlossen."
