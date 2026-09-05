#!/bin/bash
# Startet NISQA-Server + Denoise-Server im Hintergrund und danach die
# Perfect Dataset GUI. Bewusst OHNE 'set -e', damit das Fenster bei einem
# Fehler offen bleibt (wichtig bei Start per Doppelklick, sonst verschwindet
# das Terminal sofort).
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
DENOISE_PY="$SCRIPT_DIR/denoise_server/.venv/bin/python"

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

if ! "$NISQA_PY" -c "import nisqalib" 2>/dev/null; then
    echo "NISQA-venv gefunden, aber 'nisqalib' fehlt darin (oder ist nicht importierbar)."
    echo "Diagnose:"
    echo -n "  Python-Interpreter: "
    "$NISQA_PY" -c "import sys; print(sys.executable)"
    echo -n "  site-packages:      "
    "$NISQA_PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    echo "  nisqalib-Verzeichnisse dort:"
    found=$(find "$SCRIPT_DIR/nisqa_server/.venv" -maxdepth 5 -iname "*nisqalib*" 2>/dev/null)
    if [ -n "$found" ]; then
        echo "$found" | sed 's/^/    /'
    else
        echo "    (keine gefunden)"
    fi
    echo "  Fehlermeldung beim direkten Import:"
    "$NISQA_PY" -c "import nisqalib" 2>&1 | sed 's/^/    /'
    echo ""
    echo "Falls das nach 'bash nisqa_server/install.sh' weiter auftritt, bitte diese"
    echo "komplette Diagnose-Ausgabe mitschicken — dann suchen wir gezielt weiter,"
    echo "statt nochmal zu raten. Ansonsten neu installieren:"
    echo "  bash nisqa_server/install.sh"
    pause_on_error
fi

if [ ! -x "$DENOISE_PY" ]; then
    echo "Denoise-Server-venv nicht gefunden unter: $DENOISE_PY"
    echo "Der Denoise-Server braucht eine EIGENE, separate Installation:"
    echo "  bash denoise_server/install.sh"
    pause_on_error
fi

BG_PIDS=()
cleanup() {
    echo "Beende Hintergrund-Server..."
    for pid in "${BG_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

echo "Starte NISQA-Server (Port 8050)..."
(
    cd "$SCRIPT_DIR/nisqa_server"
    export LD_LIBRARY_PATH="$(torch_ld_library_path "$NISQA_PY")"
    "$NISQA_PY" -m uvicorn serve_nisqa:app --port 8050 --host 0.0.0.0
) &
BG_PIDS+=($!)

echo "Starte Denoise-Server (Port 8051)..."
(
    cd "$SCRIPT_DIR/denoise_server"
    export LD_LIBRARY_PATH="$(torch_ld_library_path "$DENOISE_PY")"
    "$DENOISE_PY" -m uvicorn serve_denoise:app --port 8051 --host 0.0.0.0
) &
BG_PIDS+=($!)

wait_for_server() {
    local name="$1" url="$2"
    echo "Warte auf $name..."
    for i in $(seq 1 30); do
        if curl -s -o /dev/null "$url"; then
            echo "$name bereit."
            return 0
        fi
        sleep 1
    done
    echo "WARNUNG: $name antwortet nach 30s nicht — GUI startet trotzdem,"
    echo "die zugehörige Funktion liefert dann aber nur Fehler/Fallback."
    return 1
}

wait_for_server "NISQA-Server" "http://localhost:8050/health"
wait_for_server "Denoise-Server" "http://localhost:8051/health"

echo "Starte Perfect Dataset GUI..."
export LD_LIBRARY_PATH="$(torch_ld_library_path "$MAIN_PY")"
"$MAIN_PY" app.py
status=$?

if [ $status -ne 0 ]; then
    pause_on_error
fi

read -p "GUI beendet. Enter drücken zum Schließen..."
