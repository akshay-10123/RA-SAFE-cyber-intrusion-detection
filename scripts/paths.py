"""Shared dataset paths for all IDS scripts."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def dataset_dir() -> Path:
    """Return the first existing dataset directory."""
    candidates = [
        PROJECT_ROOT / "data" / "dataset_updated",
        PROJECT_ROOT.parent / "dataset_updated",
        PROJECT_ROOT / "data" / "dataset_central",
        PROJECT_ROOT.parent / "dataset_central",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def dataset_central_dir() -> Path:
    """Alias used by FNN / older scripts."""
    return dataset_dir()


def standardized_csv(fusion: str) -> Path:
    return dataset_dir() / f"standardized_dataset_{fusion}.csv"


def require_dataset(fusion: str) -> Path:
    path = standardized_csv(fusion)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nRun: python scripts/prepare_dataset.py"
        )
    return path
