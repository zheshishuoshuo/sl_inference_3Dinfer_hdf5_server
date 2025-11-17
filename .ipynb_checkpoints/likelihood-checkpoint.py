"""Full 2D-kernel likelihood for 3D hyper-parameters.

Consumes LensGrid2D tables and performs the marginal likelihood integral over
(logMh, gamma_h) per lens, then aggregates across lenses and includes the
selection term A(η).

η ordering (3D): (alpha_sps, mu_h, mu_gamma)
"""

from __future__ import annotations

from typing import Iterable, Sequence, Optional, Callable

# No internal multiprocessing pools here; outer layers may parallelize walkers

import os
import numpy as np
from scipy.stats import norm, skewnorm
from .make_tabulate import LensGrid2D, tabulate_likelihood_grids
from .mock_generator.mass_sampler import MODEL_PARAMS
from .config import SCATTER
from .utils import selection_function
from .compute_A_eta import load_A_eta_interpolator, build_eta_grid




MODEL_P = MODEL_PARAMS["deVauc"]




A_INTERP: Optional[Callable[[Sequence[float]], float]] = None
# Load the new 3D A(eta) table with dynamic filename based on grid sizes
_mu_DM_grid, _mu_gamma_grid, _alpha_grid = build_eta_grid()

_fname = f"Aeta3D_merged_new_large.h5"

file_path = os.path.join(os.path.dirname(__file__), 'aeta_tables', _fname)
A_INTERP = load_A_eta_interpolator(file_path)

def A_interp(eta: Sequence[float]) -> float:
    """Evaluate A(η) interpolator at given η.

    η ordering (3D): (alpha_sps, mu_h, mu_gamma)
    """
    if A_INTERP is None:
        raise RuntimeError("A(eta) interpolator is not initialized.")

    alpha_sps, mu_h, mu_gamma = eta
    # Interpolator expects grid ordering (mu_h, mu_gamma, alpha)
    eta3 = (mu_h, mu_gamma, alpha_sps)

    return A_INTERP(eta3)

# ----------------------------------------------------------------------------
# Helpers for numerical safety
# ----------------------------------------------------------------------------
def safe_value(v: float, *, minval: float = 1e-300) -> float:
    """Return a finite scalar with a lower bound.

    - Converts NaN/inf to finite values via np.nan_to_num
    - Clamps to at least ``minval`` to protect against log(0)
    """
    vv = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return max(float(vv), minval) if minval is not None else float(vv)


# Removed K(mu1, mu2) table dependency; photometric selection and likelihood
# are handled by explicit integration over source magnitude ms per lens.

# Convenience wrapper ---------------------------------------------------------


# Priors ---------------------------------------------------------------------



def log_prior(theta: Sequence[float]) -> float:
    """Flat/top-hat prior for 3D hyperparameters η.

    η = (alpha_sps, mu_h, mu_gamma)
    """
    if len(theta) != 3:
        return -np.inf

    alpha_sps, mu_h, mu_gamma = map(float, theta)

    if not (-0.3 <= alpha_sps <= 0.3):
        return -np.inf
    if not (12.5 <= mu_h <= 13.5):
        return -np.inf
    if not (0.6 <= mu_gamma <= 1.4):
        return -np.inf

    return 0.0



    # mu_DM_grid = np.linspace(12, 14, N)
    # beta_DM_grid = np.linspace(1, 3, 15)
    # sigma_DM_grid = np.linspace(0.01, 0.6, 15)
    # mu_gamma_grid = np.linspace(0.5, 1.5, N)
    # sigma_gamma_grid = np.linspace(0.01, 0.3, N)
    # alpha_grid = np.linspace(-0.5, 0.5, N)

# def log_prior(theta: Sequence[float]) -> float:
#     """Flat/top-hat prior for 6D parameters η.

#     η = (alpha_sps, mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma)
#     """
#     if len(theta) != 6:
#         return -np.inf
#     alpha_sps, mu_h, beta_h, sigma_h, mu_gamma, sigma_gamma = map(float, theta)

#     # Broad, physically sensible ranges. Adjust in future if needed.
#     # if not (0.0 < alpha_sps < 0.3):
#     #     return -np.inf
#     # if not (11.0 < mu_h < 16.0):
#     #     return -np.inf
#     # if not (1.0 < beta_h < 5.0):
#     #     return -np.inf
#     # if not (0.2 < sigma_h < 0.6):
#     #     return -np.inf
#     # if not (0.5 < mu_gamma < 4):
#     #     return -np.inf
#     # if not (0.05 < sigma_gamma < 2):
#     #     return -np.inf
#     return 0.0

    # mu_DM_grid = np.linspace(12.6, 13.2, 30)
    # sigma_DM_grid = np.linspace(0.27, 0.47, 10)
    # alpha_grid = np.linspace(0.1, 0.2, 50)


# Single-lens integral -------------------------------------------------------


SPS_A, SPS_LOC, SPS_SCALE = (float(10 ** MODEL_P["log_s_star"]), float(MODEL_P["mu_star"]), float(MODEL_P["sigma_star"]))

def single_lens_likelihood(
    grid: LensGrid2D,
    eta: Sequence[float],
) -> float:
    """Per-lens marginal likelihood B(d_i | η) using the 2D kernel (3D η).

    B(d_i | η) = ∬ F_i(logMh, gamma_h) P(logMh | mu_h) P(gamma_h | mu_gamma) dlogMh dgamma_h
    """

    if len(eta) != 3:
        return 1e-300
    alpha_sps, mu_h, mu_gamma = map(float, eta)


    # Axes from grid
    M_halo_axis = grid.logMh_axis
    Gamma_h_axis = grid.gamma_h_axis
    
    # Vakues from grid
    logM_star_true_grid = grid.logM_star_true
    logM_sps_obs = grid.logM_star_sps_obs
    logRe_obs = grid.logRe
    sigma_star = SCATTER.star
    Msps_grid = logM_star_true_grid - alpha_sps




    # weight  


    P_Msps_obs = norm.pdf(logM_sps_obs, loc=Msps_grid, scale=sigma_star)
    P_Msps_prior = skewnorm.pdf(Msps_grid, a=SPS_A, loc=SPS_LOC, scale=SPS_SCALE)
    P_logRe = norm.pdf(logRe_obs, loc=(MODEL_P["mu_R0"] + MODEL_P["beta_R"] * (Msps_grid - 11.4)), scale=MODEL_P["sigma_R"])
    # 3D version: fixed scatters in Mh and gamma
    # P_logMh_grid = norm.pdf(M_halo_axis[:, None], loc=mu_h, scale=0.3)
    P_logMh_grid = norm.pdf(M_halo_axis[:, None], loc=(mu_h + MODEL_P["beta_h"] * (Msps_grid - 11.4)), scale=MODEL_P["sigma_h"])
    factors_constant_grid = grid.factors_constant
    P_gamma_h_grid = norm.pdf(Gamma_h_axis[None, :], loc=mu_gamma, scale=0.2)



    integrand = P_Msps_obs * P_Msps_prior * P_logRe * P_logMh_grid * factors_constant_grid * P_gamma_h_grid

    # nan to zero
    integrand = np.nan_to_num(integrand)

    I_gamma = np.trapz(integrand, x=Gamma_h_axis, axis=1)
    likelihood = np.trapz(I_gamma, x=M_halo_axis, axis=0)

    likelihood = np.nan_to_num(likelihood, nan=0.0, posinf=0.0, neginf=0.0)
    

    return likelihood




# Public API -----------------------------------------------------------------


def _worker_wrapper(args):
    g, eta = args
    return single_lens_likelihood(g, eta)
    
def log_likelihood(
    eta: Sequence[float],
    grids: Sequence[LensGrid2D],
    *,
    pool: Optional[object] = None,
) -> float:
    """Joint log-likelihood for all lenses using 2D kernels (3D η).

    logL(η) = ∑_i log B(d_i | η) − N_lens log A(η)
    """
    A_eta = A_interp(list(eta))
    if np.isnan(A_eta) or A_eta <= 0.0:
        # print(eta)
        raise ValueError("A(eta) is non-positive or NaN."+str(eta)+str(A_eta))
    # A_eta = 1



    if pool is not None and hasattr(pool, "map"):
        args_list = [(g, eta) for g in grids]  # ✅ 显式打包 eta
        results = list(pool.map(_worker_wrapper, args_list))
    else:
        results = [single_lens_likelihood(g, eta) for g in grids]


    results = np.asarray(results, dtype=float)
    logLs = np.log(results)
    total = np.sum(logLs) - len(grids) * np.log(A_eta)

    return total



def log_posterior(
    eta: Sequence[float],
    grids: Sequence[LensGrid2D],
    *,
    pool: Optional[object] = None
) -> float:
    """Posterior = prior + likelihood for 3D hyper-parameters."""

    # eta[4] = 1.0
    # eta[5] = 0.2

    lp = log_prior(eta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(eta, grids, pool=pool)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


__all__ = [
    "precompute_grids",
    "log_prior",
    "log_likelihood",
    "log_posterior",
]
