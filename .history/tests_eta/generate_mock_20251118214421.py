from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import h5py

from ..mock_generator.mock_generator import run_mock_simulation
from ..config import SCATTER  # noqa: F401  # imported for consistency / future use
from .config_test import get_mock_dir, utc_timestamp


N_LENS_DEFAULT = 1000
N


def _list_existing_mocks(index: int) -> List[Path]:
    mock_dir = get_mock_dir(index)
    return sorted(
        (p for p in mock_dir.glob("mock_*.h5") if p.is_file()),
        key=lambda p: p.name,
    )


def _extract_timestamp_from_name(name: str) -> str:
    # mock_{timestamp}.h5
    if not name.startswith("mock_") or not name.endswith(".h5"):
        return ""
    core = name[len("mock_") : -len(".h5")]
    return core


def _find_latest_mock(index: int) -> Path | None:
    mocks = _list_existing_mocks(index)
    if not mocks:
        return None
    mocks_sorted = sorted(
        mocks,
        key=lambda p: _extract_timestamp_from_name(p.name),
    )
    return mocks_sorted[-1]


def _save_mock_hdf5(
    out_path: Path,
    mock_lens_data: pd.DataFrame,
    mock_observed_data: pd.DataFrame,
    samples_dict: Dict[str, np.ndarray],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        # Lens subset (is_lensed=True)
        g_lens = f.create_group("lens")
        if not mock_lens_data.empty:
            rec = mock_lens_data.to_records(index=False)
            g_lens.create_dataset("table", data=rec, compression="gzip")

        # Observed per-lens data used for tabulation
        g_obs = f.create_group("observed")
        if not mock_observed_data.empty:
            rec_obs = mock_observed_data.to_records(index=False)
            g_obs.create_dataset("table", data=rec_obs, compression="gzip")

        # Underlying samples
        g_samples = f.create_group("samples")
        for key, arr in samples_dict.items():
            g_samples.create_dataset(key, data=np.asarray(arr), compression="gzip")


def _load_mock_hdf5(
    path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    with h5py.File(path, "r") as f:
        g_lens = f.get("lens", None)
        if g_lens is not None and "table" in g_lens:
            lens_rec = g_lens["table"][...]
            mock_lens_data = pd.DataFrame.from_records(lens_rec)
        else:
            mock_lens_data = pd.DataFrame()

        g_obs = f.get("observed", None)
        if g_obs is not None and "table" in g_obs:
            obs_rec = g_obs["table"][...]
            mock_observed_data = pd.DataFrame.from_records(obs_rec)
        else:
            mock_observed_data = pd.DataFrame()

        samples_dict: Dict[str, np.ndarray] = {}
        g_samples = f.get("samples", None)
        if g_samples is not None:
            for key in g_samples.keys():
                samples_dict[key] = np.asarray(g_samples[key][...])

    return mock_lens_data, mock_observed_data, samples_dict


def generate_or_load_mock(
    index: int,
    force_new: bool = False,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Load the latest mock for a given index or generate a new one.

    Parameters
    ----------
    index:
        Integer index identifying the mock set (1, 2, 3, ...).
    force_new:
        If True, always generate a new mock. Otherwise reuse the
        most recent mock_<timestamp>.h5 in tests_eta/data/mock_<index>/.
    """
    if not force_new:
        latest = _find_latest_mock(index)
        if latest is not None:
            mock_lens_data, mock_observed_data, samples_dict = _load_mock_hdf5(latest)
            return latest, mock_lens_data, mock_observed_data, samples_dict

    # Generate fresh mock; use index as deterministic seed
    seed = int(index)
    df_lens, mock_lens_data, mock_observed_data, samples_dict = run_mock_simulation(
        N_LENS_DEFAULT,
        logalpha=0.0,
        seed=seed,
        nbkg=4e-4,
        if_source=True,
    )
    # Keep only lensed subset in output (consistent with main)
    mock_lens_data = mock_lens_data.copy()
    mock_observed_data = mock_observed_data.copy()

    ts = utc_timestamp()
    fname = f"mock_{ts}.h5"
    mock_dir = get_mock_dir(index)
    mock_path = mock_dir / fname

    _save_mock_hdf5(mock_path, mock_lens_data, mock_observed_data, samples_dict)

    return mock_path, mock_lens_data, mock_observed_data, samples_dict
