#!/bin/bash
# Startet den NISQA-Server im Hintergrund und danach die Perfect Dataset GUI.
# Bewusst OHNE 'set -e', damit das Fenster bei einem Fehler offen bleibt
# (wichtig bei Start per Doppelklick, sonst verschwindet das Terminal sofort).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

pause_on_error() {
    echo ""
    echo "!!! Fehler — siehe Meldungen oben. !!!"
    read -p "Enter drücken zum Schließen..."
    exit 1
}

# Ermittelt die von PyTorch selbst mitgelieferten CUDA/cuDNN-Bibliotheken
# der jeweiligen venv und setzt LD_LIBRARY_PATH gezielt darauf. Ohne das
# kann eine ggf. vorhandene System-CUDA/cuDNN-Installation (für andere
# Tools) über LD_LIBRARY_PATH eine andere, zur gepinnten torch-Version
# nicht passende Version "gewinnen" -> "cuDNN version incompatibility".
torch_ld_library_path() {
    "$1" -c "
import glob, os, sysconfig
site_packages = sysconfig.get_paths()['purelib']
dirs = glob.glob(os.path.join(site_packages, 'nvidia', '*', 'lib'))
print(':'.join(dirs))
" 2>/dev/null
}

MAIN_PY="$SCRIPT_DIR/.venv/bin/python"
NISQA_PY="$SCRIPT_DIR/nisqa_server/.venv/bin/python"

if [ ! -x "$MAIN_PY" ]; then
    echo "Haupt-venv nicht gefunden unter: $MAIN_PY"
    echo "Bitte zuerst ausführen:"
    echo "  uv venv && uv pip install -r requirements.txt"
    pause_on_error
fi

if [ ! -x "$NISQA_PY" ]; then
    echo "NISQA-Server-venv nicht gefunden unter: $NISQA_PY"
    echo "Der NISQA-Server braucht eine EIGENE, separate Installation:"
    echo "  bash nisqa_server/install.sh"
    pause_on_error
fi

echo "Starte NISQA-Server (Port 8050)..."
(
    cd "$SCRIPT_DIR/nisqa_server"
    export LD_LIBRARY_PATH="$(torch_ld_library_path "$NISQA_PY")"
    "$NISQA_PY" -m uvicorn serve_nisqa:app --port 8050 --host 0.0.0.0
) &
NISQA_PID=$!

cleanup() {
    echo "Beende NISQA-Server (PID $NISQA_PID)..."
    kill "$NISQA_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Warte auf NISQA-Server..."
NISQA_READY=0
for i in $(seq 1 30); do
    if curl -s -o /dev/null http://localhost:8050/health; then
        echo "NISQA-Server bereit."
        NISQA_READY=1
        break
    fi
    sleep 1
done
if [ "$NISQA_READY" -eq 0 ]; then
    echo "WARNUNG: NISQA-Server antwortet nach 30s nicht — GUI startet trotzdem,"
    echo "NISQA-Bewertung im Review-Tab liefert dann aber nur Fehler/n.a."
fi

echo "Starte Perfect Dataset GUI..."
export LD_LIBRARY_PATH="$(torch_ld_library_path "$MAIN_PY")"
"$MAIN_PY" app.py
status=$?

if [ $status -ne 0 ]; then
    pause_on_error
fi

read -p "GUI beendet. Enter drücken zum Schließen..."
