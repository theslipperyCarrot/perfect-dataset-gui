# Perfect Dataset GUI

Lokales Gradio-Tool zur Erstellung sauberer Audio-Trainingsdatensätze für
XTTS und RVC: Import → Musik/Rauschen entfernen (Demucs + DeepFilterNet) →
VAD-Segmentierung + whisperX-Transkription → Review → Export (LJSpeech/RVC).

## Setup

```bash
uv venv && uv pip install -r requirements.txt
bash nisqa_server/install.sh
bash start.sh
```

Details zu den einzelnen Schritten: siehe Kommentare in `config.py` und `app.py`.
