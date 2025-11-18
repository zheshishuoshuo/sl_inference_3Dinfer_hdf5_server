from __future__ import annotations

from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parent / "data"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_mock_dir(index: int) -> Path:
    """Directory for a given mock index: tests_eta/data/mock_<index>."""
    return _ensure_dir(TEST_ROOT / f"mock_{int(index)}")


def get_grids_dir(index: int) -> Path:
    """Same as mock dir; grids are stored alongside mocks."""
    # Keep a single directory per index; grids distinguished by filename.
    return get_mock_dir(index)


def get_plots_dir(index: int) -> Path:
    """Directory for plots for a given index."""
    return _ensure_dir(get_mock_dir(index) / "plots")


def utc_timestamp() -> str:
    """Return an ISO-like UTC timestamp suitable for filenames."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
