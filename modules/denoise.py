"""
Optionales Denoising mit DeepFilterNet.

Demucs trennt Musik/Instrumente von Sprache, entfernt aber kein
Mikro-Hiss, Raumhall oder Lüfter-/Grundrauschen in der Sprachspur
selbst. DeepFilterNet ist ein leichtgewichtiges, echtzeitfähiges
Modell genau dafür und läuft gut lokal auf einer RTX 2080 Ti.

Wie bei Demucs (siehe demucs_separation.py) laden/speichern wir Audio
selbst über soundfile/librosa statt über DeepFilterNets eigene
load_audio()/save_audio()-Hilfsfunktionen — die nutzen intern
torchaudio.info()/load(), das aktuelle torchaudio-Versionen (>=2.9,
Umstieg auf torchcodec) entfernt haben. Die eigentliche enhance()-
Funktion braucht ohnehin nur einen fertigen Tensor, keine Datei.

DeepFilterNet arbeitet intern mit 48 kHz Mono — Ein-/Ausgabe wird
entsprechend resampled.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.torchaudio_compat import ensure_patched as ensure_torchaudio_compat

_df_model = None
_df_state = None


def _load_deepfilternet():
    """Lazy-Load, damit die GUI startet, ohne dass sofort ein Modell geladen wird."""
    global _df_model, _df_state
    if _df_model is None:
        ensure_torchaudio_compat()
        from df.enhance import init_df
        _df_model, _df_state, _ = init_df()
    return _df_model, _df_state


def denoise_file(input_path: Path, output_path: Path) -> Path:
    """Entfernt Restrauschen aus input_path und schreibt nach output_path."""
    from df.enhance import enhance

    model, df_state = _load_deepfilternet()
    target_sr = df_state.sr()

    audio, sr = sf.read(str(input_path), always_2d=True)  # (Samples, Kanäle)
    audio = audio.T.astype(np.float32)  # (Kanäle, Samples)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, axis=-1)
    if audio.shape[0] > 1:
        audio = audio.mean(axis=0, keepdims=True)  # DeepFilterNet erwartet Mono

    wav_tensor = torch.from_numpy(np.ascontiguousarray(audio))
    enhanced = enhance(model, df_state, wav_tensor)

    enhanced_np = enhanced.detach().cpu().numpy()
    if enhanced_np.ndim == 2:
        enhanced_np = enhanced_np[0]  # (Kanäle, Samples) -> (Samples,)

    sf.write(str(output_path), enhanced_np, target_sr, subtype="PCM_16")
    return output_path
