from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Sequence, Callable

import numpy as np
import h5py

from ..make_tabulate import LensGrid2D
from ..likelihood import log_likelihood as _log_likelihood_with_A
from ..likelihood import log_prior


def load_grids_from_hdf5(grids_path: Path) -> List[LensGrid2D]:
    """
    Reconstruct LensGrid2D objects from a tabulated HDF5 kernel file.
    """
    with h5py.File(grids_path, "r") as f:
        gk = f["kernel"]
        logMh_axis = np.asarray(gk["logMh_axis"][...], dtype=float)
        gamma_axis = np.asarray(gk["gamma_h_axis"][...], dtype=float)
        logM_star_true = np.asarray(gk["logM_star_true"][...], dtype=float)
        muA = np.asarray(gk["muA"][...], dtype=float)
        muB = np.asarray(gk["muB"][...], dtype=float)
        factors_constant = np.asarray(gk["factors_constant"][...], dtype=float)
        logM_star_sps_obs = np.asarray(gk["logM_star_sps_obs"][...], dtype=float)
        m1_obs = np.asarray(gk["m1_obs"][...], dtype=float)
        m2_obs = np.asarray(gk["m2_obs"][...], dtype=float)
        zl = np.asarray(gk["zl"][...], dtype=float)
        zs = np.asarray(gk["zs"][...], dtype=float)
        logRe = np.asarray(gk["logRe"][...], dtype=float)

    n_lens = logM_star_true.shape[0]
    grids: List[LensGrid2D] = []
    for i in range(n_lens):
        grid = LensGrid2D(
            logMh_axis=logMh_axis,
            gamma_h_axis=gamma_axis,
            logM_star_true=logM_star_true[i],
            muA=muA[i],
            muB=muB[i],
            logM_star_sps_obs=float(logM_star_sps_obs[i]),
            xA_obs=np.nan,  # not needed for likelihood
            xB_obs=np.nan,
            m1_obs=float(m1_obs[i]),
            m2_obs=float(m2_obs[i]),
            zl=float(zl[i]),
            zs=float(zs[i]),
            logRe=float(logRe[i]),
            lens_id=int(i),
            factors_constant=factors_constant[i],
        )
        grids.append(grid)
    return grids


def _log_likelihood_no_A(eta: Sequence[float], grids: Sequence[LensGrid2D]) -> float:
    """
    Version of log_likelihood without the A(eta) selection term.

    This mirrors likelihood.log_likelihood but omits the -N_lens log A(eta)
    factor by directly summing single_lens_likelihood.
    """
    from ..likelihood import single_lens_likelihood

    vals = [single_lens_likelihood(g, eta) for g in grids]
    arr = np.asarray(vals, dtype=float)
    arr = np.clip(arr, 1e-300, np.inf)
    return float(np.sum(np.log(arr)))


def _build_eta_vector(
    alpha_sps: float,
    mu_h: float,
    mu_gamma: float,
) -> List[float]:
    return [float(alpha_sps), float(mu_h), float(mu_gamma)]


def _select_ll_func(use_A_eta: bool) -> Callable[[Sequence[float], Sequence[LensGrid2D]], float]:
    return _log_likelihood_with_A if use_A_eta else _log_likelihood_no_A


def compute_1d_loglike(
    grids: List[LensGrid2D],
    param_name: str,
    grid_values: np.ndarray,
    fixed_params: Dict[str, float],
    *,
    use_prior: bool = True,
    use_A_eta: bool = True,
) -> np.ndarray:
    """
    Compute 1D log-likelihood or log-posterior along a given parameter.
    """
    ll_func = _select_ll_func(use_A_eta)
    out = np.empty_like(grid_values, dtype=float)

    for i, val in enumerate(grid_values):
        alpha = fixed_params.get("alpha_sps", 0.0)
        mu_h = fixed_params.get("mu_h", 12.91)
        mu_gamma = fixed_params.get("mu_gamma", 1.0)
        if param_name == "alpha_sps":
            alpha = float(val)
        elif param_name == "mu_h":
            mu_h = float(val)
        elif param_name == "mu_gamma":
            mu_gamma = float(val)
        else:
            raise ValueError(f"Unknown parameter name: {param_name}")

        eta = _build_eta_vector(alpha, mu_h, mu_gamma)
        ll = ll_func(eta, grids)
        if use_prior:
            ll += log_prior(eta)
        out[i] = ll

    out -= np.nanmax(out)
    return out


def compute_2d_loglike(
    grids: List[LensGrid2D],
    param_x: str,
    grid_x: np.ndarray,
    param_y: str,
    grid_y: np.ndarray,
    fixed_params: Dict[str, float],
    *,
    use_prior: bool = True,
    use_A_eta: bool = True,
) -> np.ndarray:
    """
    Compute 2D log-likelihood or log-posterior on a parameter plane.
    """
    ll_func = _select_ll_func(use_A_eta)
    nx = int(len(grid_x))
    ny = int(len(grid_y))
    out = np.empty((ny, nx), dtype=float)

    for j, y in enumerate(grid_y):
        for i, x in enumerate(grid_x):
            params = dict(fixed_params)
            params[param_x] = float(x)
            params[param_y] = float(y)
            eta = _build_eta_vector(
                params.get("alpha_sps", 0.0),
                params.get("mu_h", 12.9),
                params.get("mu_gamma", 1.0),
            )
            ll = ll_func(eta, grids)
            if use_prior:
                ll += log_prior(eta)
            out[j, i] = ll

    out -= np.nanmax(out)
    return out
