from __future__ import annotations

"""
Run MCMC on pre-tabulated kernels for a given mock index.

This module:
1. Loads the latest grids_<timestamp>.h5 for the specified index.
2. Loads (or generates) the corresponding mock and samples.
3. Runs emcee on the 3D hyper-parameters η = (alpha_sps, mu_h, mu_gamma).
4. Writes an HDF5 output using the same structure as the main pipeline's
   write_run_hdf5, but under tests_eta/data/mock_<index>/.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List

import emcee
import numpy as np

from .config_test import get_mock_dir, get_grids_dir, utc_timestamp
from .generate_mock import generate_or_load_mock
from .likelihood_slices import load_grids_from_hdf5
from ..config import SCATTER
from ..hdf5_io import write_run_hdf5
from ..mock_generator.mass_sampler import MODEL_PARAMS
from ..likelihood import log_posterior


def _list_existing_files(dir_path: Path, prefix: str) -> List[Path]:
    if not dir_path.exists():
        return []
    return sorted(
        (p for p in dir_path.glob(f"{prefix}_*.h5") if p.is_file()),
        key=lambda p: p.name,
    )


def _latest_grids_path(index: int) -> Path:
    grids_dir = get_grids_dir(index)
    grids = _list_existing_files(grids_dir, "grids")
    if not grids:
        raise FileNotFoundError(f"No grids_*.h5 found in {grids_dir} for index={index}")
    return grids[-1]


def run_mcmc_for_index(
    index: int,
    *,
    nwalkers: int = 40,
    nsteps: int = 1000,
    parallel: bool = False,
    nproc: int | None = None,
) -> Path:
    """
    Run MCMC on the latest tabulated grids for a given index and
    write results to an HDF5 file with the standard format.

    Parameters
    ----------
    index:
        Integer index identifying the mock set (1, 2, 3, ...).
    nwalkers, nsteps:
        MCMC configuration.
    parallel, nproc:
        Optional multiprocessing; if ``parallel`` is True and nproc is None,
        all available CPUs minus 2 are used.

    Returns
    -------
    Path
        Path to the written HDF5 run file.
    """
    # ---- 1. Load mock and observed data (or generate if missing) ----
    mock_path, mock_lens_data, mock_observed_data, samples_dict = generate_or_load_mock(
        index,
        force_new=False,
    )

    # ---- 2. Load latest grids for this index ----
    grids_path = _latest_grids_path(index)
    grids = load_grids_from_hdf5(grids_path)

    # ---- 3. Configure MCMC ----
    ndim = 3  # η = (alpha_sps, mu_h, mu_gamma)
    initial_guess = np.array([0.0, 12.9, 1.0], dtype=float)

    # Backend stored alongside mock/grids for this index
    ts = utc_timestamp()
    mock_dir = get_mock_dir(index)
    backend_file = mock_dir / f"chains_eta_index{int(index)}_{ts}.h5"
    backend_file.parent.mkdir(parents=True, exist_ok=True)

    backend = emcee.backends.HDFBackend(backend_file)
    backend.reset(nwalkers, ndim)

    # Initialize walkers around initial guess
    eps = 1e-2
    noise = eps * np.random.randn(nwalkers, ndim)
    for j in range(min(ndim, nwalkers)):
        noise[j, j] += eps
    p0 = initial_guess + noise

    # ---- 4. Run MCMC ----
    if parallel:
        import multiprocessing as mp

        if nproc is None:
            nproc = max(1, mp.cpu_count() - 2)
        with mp.Pool(processes=nproc) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                log_posterior,
                args=(grids,),
                backend=backend,
                pool=pool,
            )
            sampler.run_mcmc(p0, nsteps, progress=True)
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_posterior,
            args=(grids,),
            backend=backend,
        )
        sampler.run_mcmc(p0, nsteps, progress=True)

    # ---- 5. Collect chains and diagnostics ----
    burnin = min(200, max(0, nsteps // 5))
    chain = sampler.get_chain(discard=burnin, flat=True)
    log_prob_flat = sampler.get_log_prob(discard=burnin, flat=True)
    acceptance_frac = sampler.acceptance_fraction
    exports = {
        "samples_flat": chain,
        "log_prob": log_prob_flat,
        "acceptance_frac": acceptance_frac,
    }

    # ---- 6. Prepare metadata for write_run_hdf5 ----
    base_dir = Path(__file__).resolve().parent.parent  # project root

    # True values used in tests_eta
    model_p = MODEL_PARAMS["deVauc"]
    logalpha_true = 0.0
    true_values = [logalpha_true, float(model_p["mu_h0"]), 1.0]

    # Sample and lens counts
    sample_number = int(samples_dict.get("logM_star_sps", np.empty(0)).size)
    lens_number = int(getattr(mock_lens_data, "shape", (0,))[0])

    # Subset of samples exported (match main.py)
    subset_samples: Dict[str, Any] = {}
    for key in ("logM_star_sps", "logRe", "logM_halo"):
        if key in samples_dict:
            subset_samples[key] = samples_dict[key]

    # ---- 7. Output HDF5 run file in tests_eta/data/mock_<index> ----
    run_filename = mock_dir / f"chains_{lens_number}lens_index{int(index)}_{ts}.h5"

    write_run_hdf5(
        run_filename,
        sample_number=sample_number,
        lens_number=lens_number,
        chain_length=int(nsteps),
        scatter_mag=float(SCATTER.mag),
        scatter_star=float(SCATTER.star),
        n_galaxy=sample_number,
        eta=True,
        true_values=true_values,
        seed=int(index),
        git_root=base_dir,
        samples_dict=subset_samples,
        zl=0.3,
        zs=2.0,
        lens_table_df=mock_lens_data,
        observed_table_df=mock_observed_data,
        grids=grids,
        emcee_backend_path=backend_file,
        exports=exports,
        mock_lens_data=mock_lens_data,
    )

    return run_filename


def _build_argparser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MCMC on tests_eta grids for a given index.",
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Index of the mock set (1, 2, 3, ...).",
    )
    parser.add_argument(
        "--nwalkers",
        type=int,
        default=40,
        help="Number of MCMC walkers (default: 40).",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=1000,
        help="Number of MCMC steps (default: 1000).",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable multiprocessing for likelihood evaluation.",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="Number of processes when using --parallel.",
    )
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    out = run_mcmc_for_index(
        index=args.index,
        nwalkers=args.nwalkers,
        nsteps=args.nsteps,
        parallel=args.parallel,
        nproc=args.nproc,
    )
    print(f"[tests_eta] Wrote MCMC run file to: {out}")


if __name__ == "__main__":
    main()

