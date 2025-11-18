from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib

from .config_test import get_plots_dir


def _ensure_backend() -> None:
    if matplotlib.get_backend().lower() in {"agg", "pdf", "svg"}:
        return
    # For safety in headless environments; if already interactive, keep it.
    try:
        import os
        import sys

        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            matplotlib.use("Agg", force=True)
    except Exception:
        pass


_ensure_backend()
import matplotlib.pyplot as plt  # noqa: E402


def plot_1d_loglike(
    index: int,
    param_name: str,
    grid_values: np.ndarray,
    logL: np.ndarray,
    true_value: float | None = None,
    *,
    out_name: str | None = None,
) -> Path:
    """
    绘制 1D logL 或 exp(logL) 曲线，并保存为 png。
    """
    plots_dir = get_plots_dir(index)
    norm = np.exp(logL - np.max(logL))

    fig, ax = plt.subplots()
    ax.plot(grid_values, norm, label=param_name)
    ax.set_xlabel(param_name)
    ax.set_ylabel("relative probability")
    if true_value is not None:
        ax.axvline(true_value, color="red", linestyle="--", label="true")
    ax.legend()
    fig.tight_layout()

    if out_name is None:
        out_name = f"1d_mock_{int(index)}_{param_name}.png"
    out_path = plots_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_2d_loglike(
    index: int,
    param_x: str,
    grid_x: np.ndarray,
    param_y: str,
    grid_y: np.ndarray,
    logL_2d: np.ndarray,
    true_values: Tuple[float, float] | None = None,
    *,
    out_name: str | None = None,
) -> Path:
    """
    绘制 2D logL contour 或 imshow，并保存为 png。
    """
    plots_dir = get_plots_dir(index)
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
    fig.colorbar(im, ax=ax, label="relative probability")
    if true_values is not None:
        ax.plot(true_values[0], true_values[1], "r+", markersize=10, markeredgewidth=2)
    fig.tight_layout()

    if out_name is None:
        out_name = f"2d_mock_{int(index)}_{param_x}_{param_y}.png"
    out_path = plots_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
