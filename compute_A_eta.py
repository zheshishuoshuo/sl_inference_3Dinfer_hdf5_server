"""Monte Carlo evaluation of the normalisation factor A(eta), 3D version.

This module implements the algorithm described in the project notes for
estimating the selection-function normalisation that appears in the
likelihood.  The computation proceeds by Monte Carlo sampling of the lens
population and evaluating the integrand

    T = T1 * T2 * T3

for each sample.  Here:

* ``T1`` is the integral over source magnitude of the detection probability
  for the two lensed images weighted by the source magnitude prior.
* ``T2`` is the weighting from the random source position, proportional to
  the square of the caustic scale ``betamax`` times the uniform variate ``u``.
* ``T3`` is the (untruncated) halo–mass relation ``p(Mh | muDM, Msps)``
  evaluated at the sampled halo mass.

The final estimate of ``A`` is the average of ``T`` over all Monte Carlo
samples with an additional factor from importance sampling the halo mass with
an uninformative uniform proposal.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import h5py
from scipy.stats import norm
from tqdm import tqdm

# Support both package-relative and direct execution imports
try:
    from .config import SCATTER
    from .mock_generator.lens_model import LensModel
    from .mock_generator.lens_solver import solve_single_lens, solve_lens_parameters_from_obs
    from .mock_generator.mass_sampler import MODEL_PARAMS, generate_samples
    from .utils import selection_function
    from .build_k_table import load_K_interpolator
except ImportError:  # fallback when run as a top-level script
    from config import SCATTER
    from mock_generator.lens_model import LensModel
    from mock_generator.lens_solver import solve_single_lens, solve_lens_parameters_from_obs
    from mock_generator.mass_sampler import MODEL_PARAMS, generate_samples
    from utils import selection_function
    from build_k_table import load_K_interpolator

# -----------------------------------------------------------------------------
# Sampling utilities
# -----------------------------------------------------------------------------

# def load_K_interpolator(
#     path: str,
#     *,
#     method: Literal["linear", "nearest"] = "linear",
#     bounds_error: bool = False,
#     fill_value: float | None = 0.0,
#     return_arrays: bool = False,
# ):

MODEL_P = MODEL_PARAMS["deVauc"]

# Preload K(mu1, mu2) interpolator (integrated over source magnitude with prior)
K_interp = load_K_interpolator(
    os.path.join(os.path.dirname(__file__), "K_K_table_mu1000_ms2000.h5"),
    method="linear",
    bounds_error=False,
    fill_value=0.0,
)

def sample_lens_population(n_samples: int, zl: float = 0.3, zs: float = 2.0):
    """Draw Monte Carlo samples of the lens population.

    Parameters
    ----------
    n_samples
        Number of Monte Carlo samples to draw.
    zl, zs
        Lens and source redshifts.

    Returns
    -------
    dict
        Dictionary containing sampled stellar masses, sizes, halo masses and
        source-position variables along with the bounds of the halo-mass
        proposal distribution.
    """

    data = generate_samples(n_samples)
    logM_star_sps = data["logM_star_sps"]
    logRe = data["logRe"]
    rng = np.random.default_rng()
    gamma_in = rng.normal(loc=1.0, scale=0.5, size=n_samples)
    c_halo = data["c_halo"]
    
    beta_unit = np.random.rand(n_samples)**0.5
    # Uniform proposal for halo mass to allow importance reweighting
    logMh_min, logMh_max = 10.0, 16.0
    logMh = np.random.uniform(logMh_min, logMh_max, n_samples)

    return {
        "logM_star_sps": logM_star_sps,
        "logRe": logRe,
        "beta": beta_unit,
        "gamma_in": gamma_in,
        "logMh": logMh,
        "logMh_min": logMh_min,
        "logMh_max": logMh_max,
        "zl": zl,
        "zs": zs,
    }


# -----------------------------------------------------------------------------
# Lens-equation solver
# -----------------------------------------------------------------------------


def solve_magnification(args):
    """Solve a single lens configuration returning magnifications and caustic."""

    logM_star, logRe, logMh, gamma_in, beta_unit, zl, zs = args
    try:
        model = LensModel(
            logM_star=logM_star, logM_halo=logMh, logRe=logRe, zl=zl, zs=zs, gamma_in=gamma_in,
        )
        xA, xB = solve_single_lens(model, beta_unit)
        ycaustic = model.solve_ycaustic() or 0.0
        mu1 = model.mu_from_rt(xA)
        mu2 = model.mu_from_rt(xB)
        if not (np.isfinite(mu1) and np.isfinite(mu2) and ycaustic > 0):
            return (np.nan, np.nan, 0.0)
        return (mu1, mu2, ycaustic)
    except Exception:
        return (np.nan, np.nan, 0.0)


def compute_magnifications(
    logM_star: np.ndarray,
    logRe: np.ndarray,
    logMh: np.ndarray,
    gamma_in: np.ndarray,
    beta_unit: np.ndarray,
    zl: float,
    zs: float,
    n_jobs: int | None = None,
):
    """Compute magnifications for each Monte Carlo sample."""

    n = len(logM_star)
    args = zip(logM_star, logRe, logMh, gamma_in, beta_unit, repeat(zl), repeat(zs))
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        results = list(
            tqdm(
                pool.map(solve_magnification, args),
                total=n,
                desc="solving lenses",
                leave=False,
            )
        )
    mu1, mu2, betamax = map(np.array, zip(*results))
    mu1, mu2, betamax = np.nan_to_num(mu1, nan=np.nan), np.nan_to_num(mu2, nan=np.nan), np.nan_to_num(betamax, nan=0.0)
    return mu1, mu2, betamax


# -----------------------------------------------------------------------------
# Source magnitude prior
# -----------------------------------------------------------------------------


def ms_distribution(ms_grid: np.ndarray, alpha_s: float = -1.3, ms_star: float = 24.5):
    """Normalised PDF of the unlensed source magnitude."""

    L = 10 ** (-0.4 * (ms_grid - ms_star))
    pdf = L ** (alpha_s + 1) * np.exp(-L)
    pdf /= np.trapz(pdf, ms_grid)
    return pdf


# -----------------------------------------------------------------------------
# Main 3D A(eta) computation
# -----------------------------------------------------------------------------


def build_eta_grid():
    """Return default 3D grids for ``mu_DM``, ``mu_gamma``, ``alpha``."""

    N = 50

    mu_DM_grid = np.linspace(12.5, 13.5, N)
    mu_gamma_grid = np.linspace(0.6, 1.4, N)
    alpha_grid = np.linspace(-0.3, 0.3, N)

    return mu_DM_grid, mu_gamma_grid, alpha_grid

# 'mu_h0': 12.91,
# 'beta_h': 2.04,
# 'xi_h': 0.0,
# 'sigma_h': 0.37

def compute_A_eta(
    n_samples: int = 500000,
    ms_points: int = 200,
    m_lim: float = 26.5,
    n_jobs: int | None = None,
    *,
    alpha_grid_override: np.ndarray | None = None,
    output_suffix: str = "",
):
    """Monte Carlo estimate of the 3D normalisation grid ``A(eta)``.

    3D η = (alpha, mu_DM, mu_gamma).
    """

    samples = sample_lens_population(n_samples)

    # Source magnitude prior grid (kept for metadata; K table already folds this in)
    # ms_grid = np.linspace(20.0, 30.0, ms_points)
    # p_ms = ms_distribution(ms_grid)

    # 3D grid over (mu_DM, mu_gamma, alpha)
    (
        mu_DM_grid,
        mu_gamma_grid,
        alpha_grid_default,
    ) = build_eta_grid()

    # Optionally override alpha grid
    if alpha_grid_override is not None:
        alpha_grid = np.asarray(alpha_grid_override)
    else:
        alpha_grid = alpha_grid_default

    A_accum = np.zeros(
        (
            mu_DM_grid.size,
            mu_gamma_grid.size,
            alpha_grid.size,
        ),
        dtype=np.float64,
    )

    # Precompute p_gamma over all mu_gamma for every sample with fixed scatter 0.2
    # Shape: (N_mu_gamma, n_samples)
    gamma_samples = samples["gamma_in"]
    gamma_in = gamma_samples[None, :]
    MU_G = mu_gamma_grid[:, None]
    SG_G = 0.2
    p_gamma_table = np.exp(-0.5 * ((gamma_in - MU_G) / SG_G) ** 2) / (SG_G * np.sqrt(2 * np.pi))

    # Proposal density q_gamma and importance weights for gamma
    mu_prop = 1.0
    sigma_prop = 0.5
    q_gamma = norm.pdf(gamma_samples, loc=mu_prop, scale=sigma_prop)
    w_gamma_imp = 1.0 / q_gamma

    for a_idx, alpha in enumerate(tqdm(alpha_grid, desc="alpha loop")):
        # Mstar used in lensing is Msps + alpha
        logM_star = samples["logM_star_sps"] + alpha

        mu1, mu2, betamax = compute_magnifications(
            logM_star,
            samples["logRe"],
            samples["logMh"],
            samples["gamma_in"],
            samples["beta"],
            samples["zl"],
            samples["zs"],
            n_jobs=n_jobs,
        )



        valid = (mu1 > 0) & (mu2 > 0) & (betamax > 0)

        # Photometric/detection term integrated over source magnitude (from prebuilt K table)
        T1 = K_interp(mu1[valid], mu2[valid])

        # ---- T2: source position weighting ----
        T2 = betamax[valid] ** 2

        # Combined static weight per Monte Carlo sample
        w_static = T1 * T2

        # ---- T3: halo-mass + gamma relation for each (muDM, mu_gamma) ----
        valid_idx = np.where(valid)[0]

        # Parameters for halo-mass relation from mass_sampler (ensure consistency)
        betaDM = MODEL_P["beta_h"]
        sigmaDM = MODEL_P["sigma_h"]

        for j, i in tqdm(enumerate(valid_idx), desc="valid index loop", total=valid.sum()):
            logM_sps_i = samples["logM_star_sps"][i]
            logMh_i = samples["logMh"][i]
            w_i = float(w_static[j])
            if not (np.isfinite(w_i) and w_i > 0):
                continue

            # Vectorised halo-mass probability across (mu_DM) consistent with mass_sampler
            # mean_Mh = mu_DM + betaDM*(logM_sps - 11.4) + xiDM*(logRe - mu_r)

            mean_Mh = (
                mu_DM_grid
                + betaDM * (logM_sps_i - 11.4)
            )
            p_Mh = np.exp(-0.5 * ((logMh_i - mean_Mh) / sigmaDM) ** 2) / (sigmaDM * np.sqrt(2 * np.pi))

            # Select precomputed p_gamma for this sample i, shape (N_mu_gamma,)
            p_gamma = p_gamma_table[:, i]

            # Importance sampling weight for gamma proposal
            w_gamma_i = w_gamma_imp[i]

            # Accumulate with outer product over (mu_DM, mu_gamma)
            # Result aligns with A_accum[:, :, a_idx]
            A_accum[:, :, a_idx] += w_i * w_gamma_i * np.outer(p_Mh, p_gamma)



    Mh_range = samples["logMh_max"] - samples["logMh_min"]
    A = Mh_range * A_accum / n_samples

    # ---- Write to HDF5 (3D: mu_DM, mu_gamma, alpha) ----
    # Output to current working directory; add optional suffix to avoid overwrite
    fname = (
        f"Aeta3D_mu{mu_DM_grid.size}_mugamma{mu_gamma_grid.size}_alpha{alpha_grid.size}{output_suffix}.h5"
    )
    out_path = os.path.join(os.getcwd(), fname)

    with h5py.File(out_path, "w") as f:
        # Metadata group
        gmeta = f.create_group("metadata")
        gmeta.attrs["scatter_mag"] = float(SCATTER.mag)
        gmeta.attrs["scatter_star"] = float(SCATTER.star)
        gmeta.attrs["m_lim"] = float(m_lim)
        gmeta.attrs["alpha_s"] = float(-1.3)
        gmeta.attrs["m_s_star"] = float(24.5)
        gmeta.attrs["n_samples"] = int(n_samples)
        gmeta.attrs["K_table"] = os.path.basename(
            os.path.join(os.path.dirname(__file__), "K_mu0_10000_midRes_K_table_mu50000_ms10000.h5")
        )

        # Grids and A table (3D)
        g = f.create_group("grids")
        g.create_dataset("mu_DM_grid", data=mu_DM_grid)
        g.create_dataset("mu_gamma_grid", data=mu_gamma_grid)
        g.create_dataset("alpha_grid", data=alpha_grid)
        g.create_dataset("A_grid", data=A.astype(np.float32), compression="gzip")

        # Optional cache of Monte Carlo samples for reproducibility
        gcache = f.create_group("cache")
        gcache.create_dataset("logM_star_sps", data=samples["logM_star_sps"], compression="gzip")
        gcache.create_dataset("logRe", data=samples["logRe"], compression="gzip")
        gcache.create_dataset("logMh", data=samples["logMh"], compression="gzip")
        gcache.create_dataset("beta_unit", data=samples["beta"], compression="gzip")
        gcache.create_dataset("gamma_in", data=samples["gamma_in"], compression="gzip")
        gcache.attrs["zl"] = float(samples["zl"]) if np.isscalar(samples["zl"]) else float(np.asarray(samples["zl"]).ravel()[0])
        gcache.attrs["zs"] = float(samples["zs"]) if np.isscalar(samples["zs"]) else float(np.asarray(samples["zs"]).ravel()[0])

    return out_path


def load_A_eta_interpolator(path: str):
    """Load an interpolator for A(η) from HDF5.

    Supports 3D grid (mu_DM, mu_gamma, alpha)."""

    import h5py
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator

    with h5py.File(path, "r") as f:
        # 3D grids
        mu_DM_grid = np.array(f["grids/mu_DM_grid"]) if "grids/mu_DM_grid" in f else None
        mu_gamma_grid = np.array(f["grids/mu_gamma_grid"]) if "grids/mu_gamma_grid" in f else None
        alpha_grid = np.array(f["grids/alpha_grid"]) if "grids/alpha_grid" in f else None
        A_grid = np.array(f["grids/A_grid"]) if "grids/A_grid" in f else None

    if A_grid is None or mu_DM_grid is None or mu_gamma_grid is None or alpha_grid is None:
        raise ValueError("Missing required grids in A(eta) HDF5 file")

    if A_grid.ndim == 3:
        return RegularGridInterpolator(
            (mu_DM_grid, mu_gamma_grid, alpha_grid),
            A_grid,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    raise ValueError("Unsupported A_grid dimensionality for 3D A(eta)")


if __name__ == "__main__":
    compute_A_eta()
