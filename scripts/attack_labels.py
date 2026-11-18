"""Attack type labels for multi-class UAV IDS (T-ITS dataset row layout)."""
from __future__ import annotations

# Integer class ids (stable for training)
CLASS_BENIGN = 0
CLASS_DOS = 1
CLASS_REPLAY = 2
CLASS_EVIL_TWIN = 3
CLASS_FDI = 4

ATTACK_CLASS_NAMES: dict[int, str] = {
    CLASS_BENIGN: "benign",
    CLASS_DOS: "dos",
    CLASS_REPLAY: "replay",
    CLASS_EVIL_TWIN: "evil_twin",
    CLASS_FDI: "fdi",
}

ATTACK_DISPLAY_NAMES: dict[int, str] = {
    CLASS_BENIGN: "Benign (normal flight)",
    CLASS_DOS: "De-authentication DoS",
    CLASS_REPLAY: "Replay attack",
    CLASS_EVIL_TWIN: "Evil twin",
    CLASS_FDI: "False data injection (FDI)",
}

NUM_ATTACK_CLASSES = len(ATTACK_CLASS_NAMES)


def normalize_class_string(value: object) -> str | None:
    """Map raw CSV class strings to canonical names."""
    if value is None or (isinstance(value, float) and str(value) == "nan"):
        return None
    text = str(value).strip().lower()
    if not text or text == "class" or text == "nan":
        return None
    if text in ("benign", "normal", "0"):
        return "benign"
    if "dos" in text or "deauth" in text or "de-auth" in text:
        return "dos"
    if "replay" in text:
        return "replay"
    if "evil" in text and "twin" in text:
        return "evil_twin"
    if "fdi" in text or "false" in text and "inject" in text:
        return "fdi"
    if text == "attack":
        return None
    return None


def canonical_to_id(name: str) -> int:
    mapping = {v: k for k, v in ATTACK_CLASS_NAMES.items()}
    return mapping[name]


def label_from_row_index(index: int) -> int:
    """
    Row ranges from the public T-ITS dataset (0-based indices).

    See: https://github.com/uamughal/UAVs-Dataset-Under-Normal-and-Cyberattacks
    """
    if index < 13717:
        return CLASS_BENIGN
    if index <= 26362:
        return CLASS_DOS
    if index <= 39343:
        return CLASS_REPLAY
    if index <= 50501:
        return CLASS_EVIL_TWIN
    return CLASS_FDI


def resolve_attack_class(raw_class: object, row_index: int) -> tuple[int, str]:
    """Prefer explicit CSV label; fall back to paper row segments."""
    parsed = normalize_class_string(raw_class)
    if parsed is not None:
        class_id = canonical_to_id(parsed)
        return class_id, ATTACK_CLASS_NAMES[class_id]
    class_id = label_from_row_index(row_index)
    return class_id, ATTACK_CLASS_NAMES[class_id]
