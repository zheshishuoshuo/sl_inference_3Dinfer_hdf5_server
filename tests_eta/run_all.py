from __future__ import annotations

import argparse

import numpy as np

from .generate_mock import generate_or_load_mock
from .tabulate_mock import tabulate_for_mock
from .likelihood_slices import (
    load_grids_from_hdf5,
    compute_1d_loglike,
    compute_2d_loglike,
)
from .plot_slices import plot_1d_loglike, plot_2d_loglike
from ..mock_generator.mass_sampler import MODEL_PARAMS


def run_test_pipeline(
    index: int,
    *,
    force_new_mock: bool = False,
) -> None:
    """
    End-to-end injection–recovery test for (alpha_sps, mu_h, mu_gamma).

    Parameters
    ----------
    index:
        Integer index identifying the mock set (1, 2, 3, ...).
    force_new_mock:
        If True, force generating a new mock instead of reusing the latest one.
    """
    # 1. Generate or load mock lenses
    mock_path, mock_lens_data, mock_observed_data, samples = generate_or_load_mock(
        index,
        force_new=force_new_mock,
    )

    # 2. Tabulate kernels for this mock
    grids_path, grids = tabulate_for_mock(
        index,
        mock_path,
    )

    # 3. Optionally reload grids from disk (to test I/O)
    grids = load_grids_from_hdf5(grids_path)

    # Ground truth for injection
    alpha_true = 0.0
    model_p = MODEL_PARAMS["deVauc"]
    mu_h_true = float(model_p["mu_h0"])
    mu_gamma_true = 1.0

    # 1D grids around true values
    alpha_grid = np.linspace(-0.1, 0.1, 80)
    mu_h_grid = np.linspace(mu_h_true - 0.3, mu_h_true + 0.3, 80)
    mu_gamma_grid = np.linspace(0.6, 1.4, 80)

    fixed = {
        "alpha_sps": alpha_true,
        "mu_h": mu_h_true,
        "mu_gamma": mu_gamma_true,
    }

    # 4. Compute and plot 1D slices
    logL_alpha = compute_1d_loglike(
        grids,
        "alpha_sps",
        alpha_grid,
        fixed_params=fixed,
        use_prior=True,
        use_A_eta=True,
    )
    plot_1d_loglike(
        index,
        "alpha_sps",
        alpha_grid,
        logL_alpha,
        true_value=alpha_true,
    )

    logL_mu_h = compute_1d_loglike(
        grids,
        "mu_h",
        mu_h_grid,
        fixed_params=fixed,
        use_prior=True,
        use_A_eta=True,
    )
    plot_1d_loglike(
        index,
        "mu_h",
        mu_h_grid,
        logL_mu_h,
        true_value=mu_h_true,
    )

    logL_mu_gamma = compute_1d_loglike(
        grids,
        "mu_gamma",
        mu_gamma_grid,
        fixed_params=fixed,
        use_prior=True,
        use_A_eta=True,
    )
    plot_1d_loglike(
        index,
        "mu_gamma",
        mu_gamma_grid,
        logL_mu_gamma,
        true_value=mu_gamma_true,
    )

    # 5. 2D slice over (mu_h, mu_gamma) at fixed alpha_sps
    grid_x = mu_h_grid
    grid_y = mu_gamma_grid
    logL_2d = compute_2d_loglike(
        grids,
        "mu_h",
        grid_x,
        "mu_gamma",
        grid_y,
        fixed_params={"alpha_sps": alpha_true},
        use_prior=True,
        use_A_eta=True,
    )
    plot_2d_loglike(
        index,
        "mu_h",
        grid_x,
        "mu_gamma",
        grid_y,
        logL_2d,
        true_values=(mu_h_true, mu_gamma_true),
    )

    # Simple diagnostics: print argmax locations
    def _argmax_1d(grid: np.ndarray, logL: np.ndarray) -> float:
        return float(grid[np.argmax(logL)])

    alpha_best = _argmax_1d(alpha_grid, logL_alpha)
    mu_h_best = _argmax_1d(mu_h_grid, logL_mu_h)
    mu_gamma_best = _argmax_1d(mu_gamma_grid, logL_mu_gamma)

    j_best, i_best = np.unravel_index(np.argmax(logL_2d), logL_2d.shape)
    mu_h_best_2d = float(grid_x[i_best])
    mu_gamma_best_2d = float(grid_y[j_best])

    print(f"[mock index {index}] 1D maxima:")
    print(f"  alpha_sps: best={alpha_best:.4f}, true={alpha_true:.4f}")
    print(f"  mu_h     : best={mu_h_best:.4f}, true={mu_h_true:.4f}")
    print(f"  mu_gamma : best={mu_gamma_best:.4f}, true={mu_gamma_true:.4f}")
    print(f"[mock index {index}] 2D (mu_h, mu_gamma) maximum:")
    print(f"  mu_h     : best={mu_h_best_2d:.4f}, true={mu_h_true:.4f}")
    print(f"  mu_gamma : best={mu_gamma_best_2d:.4f}, true={mu_gamma_true:.4f}")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run injection–recovery test for a given mock index.",
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Index of the mock set (1, 2, 3, ...).",
    )
    parser.add_argument(
        "--force-new-mock",
        action="store_true",
        help="Force regeneration of mock rather than reusing the latest one.",
    )
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    run_test_pipeline(index=args.index, force_new_mock=args.force_new_mock)


if __name__ == "__main__":
    main()
