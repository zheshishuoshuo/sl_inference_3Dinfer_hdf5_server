from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import matplotlib

from .likelihood_slices import (
    load_grids_from_hdf5,
    compute_1d_loglike,
    compute_2d_loglike,
)
from ..mock_generator.mass_sampler import MODEL_PARAMS


def _ensure_backend() -> None:
    """
    Ensure a non-interactive matplotlib backend in headless environments.
    """
    if matplotlib.get_backend().lower() in {"agg", "pdf", "svg"}:
        return
    try:
        import os
        import sys

        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            matplotlib.use("Agg", force=True)
    except Exception:
        # Fall back silently if anything goes wrong.
        pass


_ensure_backend()
import matplotlib.pyplot as plt  # noqa: E402


def timestamp() -> str:
    """
    Generate a UTC timestamp string suitable for file name suffixes.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%SZ")


@dataclass
class GridConfig:
    alpha_grid: np.ndarray
    mu_h_grid: np.ndarray
    mu_gamma_grid: np.ndarray
    alpha_true: float
    mu_h_true: float
    mu_gamma_true: float


def write_grid_config(outdir: Path, cfg: GridConfig, ts: str) -> Path:
    """
    Record the grids used to a text file.

    For each parameter:
      - If scanning:  'param_name  min  max  N'
      - If fixed:     'param_name  value'

    In this script all three parameters are scanned in 1D, so they are recorded
    as scanning grids.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"grid_config_{ts}.txt"

    lines: List[str] = []
    # alpha_sps grid (scanning)
    lines.append(
        f"alpha_sps  {cfg.alpha_grid.min():.8g}  "
        f"{cfg.alpha_grid.max():.8g}  {cfg.alpha_grid.size}"
    )
    # mu_h grid (scanning)
    lines.append(
        f"mu_h  {cfg.mu_h_grid.min():.8g}  "
        f"{cfg.mu_h_grid.max():.8g}  {cfg.mu_h_grid.size}"
    )
    # mu_gamma grid (scanning)
    lines.append(
        f"mu_gamma  {cfg.mu_gamma_grid.min():.8g}  "
        f"{cfg.mu_gamma_grid.max():.8g}  {cfg.mu_gamma_grid.size}"
    )

    content = "\n".join(lines) + "\n"
    path.write_text(content, encoding="utf-8")
    return path


def save_1d_plot(
    param_name: str,
    grid_values: np.ndarray,
    logL: np.ndarray,
    *,
    true_value: float | None,
    outdir: Path,
    ts: str,
) -> Path:
    """
    Save a 1D posterior curve (normalized exp(logL)) as a PNG.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    norm = np.exp(logL - np.max(logL))

    fig, ax = plt.subplots()
    ax.plot(grid_values, norm, label=param_name)
    ax.set_xlabel(param_name)
    ax.set_ylabel("relative posterior")
    if true_value is not None:
        ax.axvline(true_value, color="red", linestyle="--", label="true")
    ax.legend()
    fig.tight_layout()

    out_name = f"posterior_1d_{param_name}_{ts}.png"
    out_path = outdir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_2d_plot(
    param_x: str,
    grid_x: np.ndarray,
    param_y: str,
    grid_y: np.ndarray,
    logL_2d: np.ndarray,
    *,
    true_values: Tuple[float, float] | None,
    outdir: Path,
    ts: str,
) -> Path:
    """
    Save a 2D posterior contour (as an image of exp(logL)) as a PNG.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    norm = np.exp(logL_2d - np.max(logL_2d))

    fig, ax = plt.subplots()
    extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
    im = ax.imshow(
        norm,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    fig.colorbar(im, ax=ax, label="relative posterior")
    if true_values is not None:
        ax.plot(true_values[0], true_values[1], "r+", markersize=10, markeredgewidth=2)
    fig.tight_layout()

    out_name = f"posterior_2d_{param_x}_{param_y}_{ts}.png"
    out_path = outdir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_posterior(grids_path: Path, outdir: Path) -> Dict[str, Path]:
    """
    Load tabulated grids, compute posterior slices, and save results.

    Parameters
    ----------
    grids_path:
        Path to an existing tabulated HDF5 kernel file.
    outdir:
        Directory in which to store .npy results, plots, and grid config.
    """
    outdir = outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ts = timestamp()

    # Load pre-tabulated grids; do not generate new mocks or tables.
    grids = load_grids_from_hdf5(grids_path)

    # Truth values from MODEL_PARAMS["deVauc"].
    alpha_true = 0.0
    model_p = MODEL_PARAMS["deVauc"]
    mu_h_true = float(model_p["mu_h0"])
    mu_gamma_true = 1.0

    # Hyperparameter grids
    alpha_grid = np.linspace(-0.1, 0.1, 120)
    mu_h_grid = np.linspace(mu_h_true - 0.3, mu_h_true + 0.3, 120)
    mu_gamma_grid = np.linspace(0.6, 1.4, 120)

    cfg = GridConfig(
        alpha_grid=alpha_grid,
        mu_h_grid=mu_h_grid,
        mu_gamma_grid=mu_gamma_grid,
        alpha_true=alpha_true,
        mu_h_true=mu_h_true,
        mu_gamma_true=mu_gamma_true,
    )
    write_grid_config(outdir, cfg, ts)

    fixed_params = {
        "alpha_sps": alpha_true,
        "mu_h": mu_h_true,
        "mu_gamma": mu_gamma_true,
    }

    # 1D posterior slices with A(eta) computed on the fly.
    logL_alpha = compute_1d_loglike(
        grids,
        "alpha_sps",
        alpha_grid,
        fixed_params=fixed_params,
        use_prior=True,
        use_A_eta=True,
        use_A_eta_table=False,
    )
    logL_mu_h = compute_1d_loglike(
        grids,
        "mu_h",
        mu_h_grid,
        fixed_params=fixed_params,
        use_prior=True,
        use_A_eta=True,
        use_A_eta_table=False,
    )
    logL_mu_gamma = compute_1d_loglike(
        grids,
        "mu_gamma",
        mu_gamma_grid,
        fixed_params=fixed_params,
        use_prior=True,
        use_A_eta=True,
        use_A_eta_table=False,
    )

    # 2D posterior slice over (mu_h, mu_gamma) at fixed alpha_sps.
    logL_2d_mu_h_mu_gamma = compute_2d_loglike(
        grids,
        "mu_h",
        mu_h_grid,
        "mu_gamma",
        mu_gamma_grid,
        fixed_params={"alpha_sps": alpha_true},
        use_prior=True,
        use_A_eta=True,
        use_A_eta_table=False,
    )

    # Save numerical results as .npy
    out_paths: Dict[str, Path] = {}

    # 1D arrays
    out_paths["alpha_grid"] = outdir / f"grid_alpha_sps_{ts}.npy"
    np.save(out_paths["alpha_grid"], alpha_grid)
    out_paths["mu_h_grid"] = outdir / f"grid_mu_h_{ts}.npy"
    np.save(out_paths["mu_h_grid"], mu_h_grid)
    out_paths["mu_gamma_grid"] = outdir / f"grid_mu_gamma_{ts}.npy"
    np.save(out_paths["mu_gamma_grid"], mu_gamma_grid)

    out_paths["logL_alpha"] = outdir / f"posterior_1d_alpha_sps_{ts}.npy"
    np.save(out_paths["logL_alpha"], logL_alpha)
    out_paths["logL_mu_h"] = outdir / f"posterior_1d_mu_h_{ts}.npy"
    np.save(out_paths["logL_mu_h"], logL_mu_h)
    out_paths["logL_mu_gamma"] = outdir / f"posterior_1d_mu_gamma_{ts}.npy"
    np.save(out_paths["logL_mu_gamma"], logL_mu_gamma)

    # 2D array
    out_paths["logL_2d_mu_h_mu_gamma"] = (
        outdir / f"posterior_2d_mu_h_mu_gamma_{ts}.npy"
    )
    np.save(out_paths["logL_2d_mu_h_mu_gamma"], logL_2d_mu_h_mu_gamma)

    # Save plots
    out_paths["plot_1d_alpha"] = save_1d_plot(
        "alpha_sps",
        alpha_grid,
        logL_alpha,
        true_value=alpha_true,
        outdir=outdir,
        ts=ts,
    )
    out_paths["plot_1d_mu_h"] = save_1d_plot(
        "mu_h",
        mu_h_grid,
        logL_mu_h,
        true_value=mu_h_true,
        outdir=outdir,
        ts=ts,
    )
    out_paths["plot_1d_mu_gamma"] = save_1d_plot(
        "mu_gamma",
        mu_gamma_grid,
        logL_mu_gamma,
        true_value=mu_gamma_true,
        outdir=outdir,
        ts=ts,
    )
    out_paths["plot_2d_mu_h_mu_gamma"] = save_2d_plot(
        "mu_h",
        mu_h_grid,
        "mu_gamma",
        mu_gamma_grid,
        logL_2d_mu_h_mu_gamma,
        true_values=(mu_h_true, mu_gamma_true),
        outdir=outdir,
        ts=ts,
    )

    return out_paths


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute 1D and 2D posterior slices over "
            "(alpha_sps, mu_h, mu_gamma) using pre-tabulated grids."
        ),
    )
    parser.add_argument(
        "--grids",
        type=str,
        required=True,
        help="Path to existing tabulated grids HDF5 file (grids_xxx.h5).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for .npy, .png, and grid_config files.",
    )
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    grids_path = Path(args.grids).expanduser().resolve()
    outdir = Path(args.outdir)

    run_posterior(grids_path=grids_path, outdir=outdir)


if __name__ == "__main__":
    main()

