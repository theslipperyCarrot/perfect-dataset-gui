"""
Review & Korrektur: einzelne Clips anhören, Transkript korrigieren,
freigeben oder verwerfen. Arbeitet direkt auf dem manifest.json aus
dem Segmentierungs-/Transkriptionsschritt.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SEGMENTS_DIR
from modules.segment_and_transcribe import ClipEntry, load_manifest, save_manifest


def get_clip_audio_path(entry: ClipEntry) -> str:
    return str(SEGMENTS_DIR / entry.clip_filename)


def update_clip(entries: list[ClipEntry], clip_id: str, text: str = None, approved: bool = None) -> list[ClipEntry]:
    for e in entries:
        if e.id == clip_id:
            if text is not None:
                e.text = text
            if approved is not None:
                e.approved = approved
            break
    save_manifest(entries)
    return entries


def review_counts(entries: list[ClipEntry]) -> tuple[int, int, int, int]:
    total = len(entries)
    approved = sum(1 for e in entries if e.approved is True)
    rejected = sum(1 for e in entries if e.approved is False)
    pending = total - approved - rejected
    return total, approved, rejected, pending
