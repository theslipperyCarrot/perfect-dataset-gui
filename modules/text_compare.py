"""
Gemeinsame Textvergleichs-Funktionen — von modules/roundtrip_check.py und
modules/segment_and_transcribe.py (reclip_entry) genutzt. Eigenes, von
beiden abhängigkeitsfreies Modul, damit keine Zirkel-Importe zwischen den
beiden entstehen.
"""
import re


def normalize_word(w: str) -> str:
    return re.sub(r"[^\w]", "", w, flags=re.UNICODE).lower()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Klassische Editierdistanz auf Wortebene (Einfügen/Löschen/Ersetzen je
    1 Wort), normiert auf die Wortzahl des Referenztexts. Satzzeichen/Groß-
    Kleinschreibung werden beim Vergleich ignoriert — uns interessiert, ob
    die WÖRTER stimmen, nicht die exakte Schreibweise."""
    ref = [w for w in (normalize_word(w) for w in reference.split()) if w]
    hyp = [w for w in (normalize_word(w) for w in hypothesis.split()) if w]

    if not ref:
        return 0.0 if not hyp else 1.0

    prev_row = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        curr_row = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr_row[j] = min(
                prev_row[j] + 1,        # Löschen
                curr_row[j - 1] + 1,    # Einfügen
                prev_row[j - 1] + cost,  # Ersetzen/Treffer
            )
        prev_row = curr_row

    return prev_row[len(hyp)] / len(ref)
