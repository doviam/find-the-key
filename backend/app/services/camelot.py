"""Musical key ↔ Camelot mapping and harmonic compatibility (DJ mixing)."""

from __future__ import annotations

# Canonical key strings (lowercase roots, "major" / "minor") → Camelot code
# Aligned with common Mixed In Key / Camelot wheel numbering.
_KEY_TO_CAMELOT: dict[str, str] = {
    "b major": "1B",
    "f# major": "2B",
    "gb major": "2B",
    "db major": "3B",
    "c# major": "3B",
    "ab major": "4B",
    "g# major": "4B",
    "eb major": "5B",
    "d# major": "5B",
    "bb major": "6B",
    "a# major": "6B",
    "f major": "7B",
    "c major": "8B",
    "g major": "9B",
    "d major": "10B",
    "a major": "11B",
    "e major": "12B",
    "ab minor": "1A",
    "g# minor": "1A",
    "eb minor": "2A",
    "d# minor": "2A",
    "bb minor": "3A",
    "a# minor": "3A",
    "f minor": "4A",
    "c minor": "5A",
    "g minor": "6A",
    "d minor": "7A",
    "a minor": "8A",
    "e minor": "9A",
    "b minor": "10A",
    "f# minor": "11A",
    "gb minor": "11A",
    "c# minor": "12A",
    "db minor": "12A",
}


def normalize_key_name(key: str) -> str:
    s = key.strip().lower().replace("♯", "#").replace("♭", "b")
    s = " ".join(s.split())
    return s


def key_to_camelot(key: str) -> str | None:
    """Map a key string like 'C minor' or 'Db major' to Camelot code, or None if unknown."""
    n = normalize_key_name(key)
    if n in _KEY_TO_CAMELOT:
        return _KEY_TO_CAMELOT[n]
    # Try without enharmonic duplicates already covered
    return None


def camelot_compatible(camelot: str) -> list[str]:
    """
    Harmonic mixing neighbors: same slot, ±1 on the wheel (same mode),
    and relative major/minor (same number, flip A/B).
    Order: perfect, relative, +1, -1 (deduplicated).
    """
    code = camelot.strip().upper()
    if len(code) < 2 or code[-1] not in ("A", "B"):
        return [camelot]

    num = int(code[:-1])
    letter = code[-1]
    if not 1 <= num <= 12:
        return [camelot]

    def wrap(n: int) -> int:
        if n < 1:
            return 12
        if n > 12:
            return 1
        return n

    relative_letter = "B" if letter == "A" else "A"
    out: list[str] = [
        f"{num}{letter}",
        f"{num}{relative_letter}",
        f"{wrap(num + 1)}{letter}",
        f"{wrap(num - 1)}{letter}",
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    return unique
