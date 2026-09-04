"""
Kompatibilitäts-Shim für torchaudio >=2.9 (Umstieg auf torchcodec, altes
Lade-/Backend-System komplett entfernt).

Mehrere unserer Abhängigkeiten (demucs-torchcodec, DeepFilterNet, Silero
VAD, whisperX/pyannote.audio) sind selbst nicht mehr aktiv genug gepflegt,
um mit den entfernten APIs umzugehen — sie referenzieren torchaudio.backend,
torchaudio.AudioMetaData, torchaudio.list_audio_backends() usw., die es in
aktuellen torchaudio-Versionen nicht mehr gibt.

Statt bei jedem neu auftauchenden fehlenden Attribut einzeln nachzubessern
(siehe Git-Historie...), patchen wir hier EINMAL zentral eine funktionierende
Kompatibilitätsschicht — echt über soundfile/librosa implementiert, keine
leeren Attrappen. Das ist bewusst so gewählt: sollte eine dieser Bibliotheken
tatsächlich mal torchaudio.load()/save()/info() für echte Arbeit aufrufen
(nicht nur als toter Typ-Hint), bekommt sie ein echtes, korrektes Ergebnis
statt still falscher/leerer Daten.

Verwendung: ensure_patched() einmal aufrufen, BEVOR eine der betroffenen
Bibliotheken importiert wird (demucs, df.enhance, whisperx/pyannote).
Idempotent — mehrfacher Aufruf ist unproblematisch.
"""
import sys
import types
import importlib.abc
import importlib.machinery

import numpy as np
import soundfile as sf

_PATCHED = False


# ---------------------------------------------------------------------
# torchaudio.backend.* (Submodul-Importe) — permissiver Platzhalter, da wir
# nie wirklich etwas daraus aufrufen (nur tote Importe abfangen).
# ---------------------------------------------------------------------

class _DummyAny:
    def __call__(self, *a, **k):
        return _DummyAny()

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _DummyAny()


class _PermissiveModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _DummyAny()


class _TorchaudioBackendShimLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = _PermissiveModule(spec.name)
        mod.__path__ = []
        return mod

    def exec_module(self, module):
        pass


class _TorchaudioBackendShimFinder(importlib.abc.MetaPathFinder):
    PREFIX = "torchaudio.backend"

    def find_spec(self, fullname, path, target=None):
        if fullname == self.PREFIX or fullname.startswith(self.PREFIX + "."):
            return importlib.machinery.ModuleSpec(
                fullname, _TorchaudioBackendShimLoader(), is_package=True,
            )
        return None


# ---------------------------------------------------------------------
# torchaudio.AudioMetaData / info() / load() / save() / *_audio_backend() —
# ECHTE, über soundfile funktionierende Implementierungen.
# ---------------------------------------------------------------------

class AudioMetaData:
    def __init__(self, sample_rate=0, num_frames=0, num_channels=0,
                 bits_per_sample=0, encoding=""):
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.bits_per_sample = bits_per_sample
        self.encoding = encoding


def _shim_info(filepath, **kwargs):
    info = sf.info(str(filepath))
    return AudioMetaData(
        sample_rate=info.samplerate, num_frames=info.frames,
        num_channels=info.channels, bits_per_sample=0,
        encoding=info.subtype or "",
    )


def _shim_load(filepath, **kwargs):
    import torch
    audio, sr = sf.read(str(filepath), always_2d=True, dtype="float32")
    audio = audio.T  # (Samples, Kanäle) -> (Kanäle, Samples), wie torchaudio.load()
    return torch.from_numpy(np.ascontiguousarray(audio)), sr


def _shim_save(filepath, src, sample_rate, **kwargs):
    audio = src.detach().cpu().numpy()
    if audio.ndim == 2:
        audio = audio.T  # (Kanäle, Samples) -> (Samples, Kanäle) für soundfile
    sf.write(str(filepath), audio, sample_rate)


def _patch_torch_load_weights_only_default():
    """
    PyTorch >=2.6 hat den Default von `weights_only` in torch.load() von
    False auf True umgestellt (Sicherheitsmaßnahme gegen Pickle-basierte
    Codeausführung aus nicht vertrauenswürdigen Checkpoints). Ältere
    Checkpoint-Dateien wie Sileros/pyannotes VAD-Modell wurden noch mit dem
    alten Pickle-Format gespeichert (u.a. mit omegaconf.ListConfig) und
    scheitern dadurch beim Laden.

    Da alle hier geladenen Modelle aus offiziellen, vertrauenswürdigen
    Quellen kommen (silero-vad, whisperx/pyannote, demucs, DeepFilterNet,
    NISQA — keine beliebigen/unbekannten Checkpoints), stellen wir das alte
    Verhalten (weights_only=False) hier bewusst wieder her, statt bei jedem
    neu auftauchenden "Unsupported global"-Fehler eine weitere Klasse
    einzeln freizuschalten.
    """
    import torch
    if getattr(torch.load, "_pdg_patched", False):
        return
    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        # Erzwingen statt nur vorbelegen: manche Bibliotheken (z.B.
        # pytorch-lightning, das pyannote intern nutzt) übergeben
        # weights_only=True bereits EXPLIZIT — setdefault() würde da nichts
        # bewirken, da der Wert schon gesetzt ist. Wir überschreiben deshalb
        # bewusst hart, denn der ganze Zweck dieses Patches ist "in diesem
        # Tool sind alle geladenen Checkpoints vertrauenswürdig".
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    _patched_load._pdg_patched = True
    torch.load = _patched_load


def ensure_patched():
    """Patcht das echte torchaudio-Modul einmalig mit einer funktionierenden
    Kompatibilitätsschicht für in aktuellen Versionen entfernte APIs."""
    global _PATCHED
    if _PATCHED:
        return
    import torchaudio

    _patch_torch_load_weights_only_default()

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = AudioMetaData
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda name=None: None
    if not hasattr(torchaudio, "info"):
        torchaudio.info = _shim_info
    if not hasattr(torchaudio, "load"):
        torchaudio.load = _shim_load
    if not hasattr(torchaudio, "save"):
        torchaudio.save = _shim_save

    try:
        import torchaudio.backend  # noqa: F401
    except ModuleNotFoundError:
        if not any(isinstance(f, _TorchaudioBackendShimFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, _TorchaudioBackendShimFinder())

    _PATCHED = True
