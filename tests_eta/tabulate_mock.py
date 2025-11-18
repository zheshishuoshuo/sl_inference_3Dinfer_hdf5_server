from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import h5py
import numpy as np
import pandas as pd

from ..make_tabulate import tabulate_likelihood_grids, LensGrid2D
from ..main import DM_GRID_2D
from .config_test import get_grids_dir, utc_timestamp


def load_mock_observed(mock_path: Path) -> pd.DataFrame:
    """
    从 generate_or_load_mock 保存的 HDF5 中读取 mock_observed_data。
    """
    with h5py.File(mock_path, "r") as f:
        g_obs = f["observed"]
        rec = g_obs["table"][...]
    df = pd.DataFrame.from_records(rec)
    return df


def _save_grids_hdf5(
    out_path: Path,
    grids: List[LensGrid2D],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not grids:
        raise ValueError("No grids to save.")

    n_lens = len(grids)
    logMh_axis = grids[0].logMh_axis
    gamma_axis = grids[0].gamma_h_axis
    n_Mh = logMh_axis.size
    n_gamma = gamma_axis.size

    logM_star_true = np.empty((n_lens, n_Mh, n_gamma), dtype=float)
    muA = np.empty((n_lens, n_Mh, n_gamma), dtype=float)
    muB = np.empty((n_lens, n_Mh, n_gamma), dtype=float)
    factors_constant = np.empty((n_lens, n_Mh, n_gamma), dtype=float)
    logM_star_sps_obs = np.empty((n_lens,), dtype=float)
    m1_obs = np.empty((n_lens,), dtype=float)
    m2_obs = np.empty((n_lens,), dtype=float)
    zl = np.empty((n_lens,), dtype=float)
    zs = np.empty((n_lens,), dtype=float)
    logRe = np.empty((n_lens,), dtype=float)

    for i, g in enumerate(grids):
        logM_star_true[i] = g.logM_star_true
        muA[i] = g.muA
        muB[i] = g.muB
        factors_constant[i] = g.factors_constant
        logM_star_sps_obs[i] = g.logM_star_sps_obs
        m1_obs[i] = g.m1_obs
        m2_obs[i] = g.m2_obs
        zl[i] = g.zl
        zs[i] = g.zs
        logRe[i] = g.logRe

    with h5py.File(out_path, "w") as f:
        gk = f.create_group("kernel")
        gk.create_dataset("logMh_axis", data=logMh_axis, compression="gzip")
        gk.create_dataset("gamma_h_axis", data=gamma_axis, compression="gzip")
        gk.create_dataset("logM_star_true", data=logM_star_true, compression="gzip")
        gk.create_dataset("muA", data=muA, compression="gzip")
        gk.create_dataset("muB", data=muB, compression="gzip")
        gk.create_dataset("factors_constant", data=factors_constant, compression="gzip")
        gk.create_dataset("logM_star_sps_obs", data=logM_star_sps_obs, compression="gzip")
        gk.create_dataset("m1_obs", data=m1_obs, compression="gzip")
        gk.create_dataset("m2_obs", data=m2_obs, compression="gzip")
        gk.create_dataset("zl", data=zl, compression="gzip")
        gk.create_dataset("zs", data=zs, compression="gzip")
        gk.create_dataset("logRe", data=logRe, compression="gzip")


def tabulate_for_mock(
    index: int,
    mock_path: Path,
) -> Tuple[Path, List[LensGrid2D]]:
    """
    对指定 mock 做 tabulate 并保存到 index 对应目录。
    """
    mock_observed_data = load_mock_observed(mock_path)
    dm_grid = DM_GRID_2D

    grids = tabulate_likelihood_grids(
        mock_observed_data,
        dm_grid,
        n_jobs=None,
    )

    logMh_axis = np.asarray(dm_grid.logMh, dtype=float)
    gamma_axis = np.asarray(dm_grid.gamma_h, dtype=float)
    ts = utc_timestamp()
    fname = f"grids_{ts}.h5"
    grids_dir = get_grids_dir(index)
    out_path = grids_dir / fname
    _save_grids_hdf5(out_path, grids)
    return out_path, grids
