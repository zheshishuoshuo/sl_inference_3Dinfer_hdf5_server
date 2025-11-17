import os
import sys
import argparse
import numpy as np


def _load_module():
    """Import compute_A_eta and build_eta_grid with robust package handling.

    Supports running this script directly from the repository root without
    relying on `-m package` invocation.
    """
    try:
        # When package-relative imports in compute_A_eta.py are valid
        from compute_A_eta import compute_A_eta, build_eta_grid  # type: ignore
        return compute_A_eta, build_eta_grid
    except Exception:
        # Try importing via package name if running from outside
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        parent = os.path.dirname(pkg_dir)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        pkg_name = os.path.basename(pkg_dir)
        mod_name = f"{pkg_name}.compute_A_eta"
        import importlib

        mod = importlib.import_module(mod_name)
        return mod.compute_A_eta, mod.build_eta_grid


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a part of A(eta) grid computation on one node.")
    parser.add_argument("part_id", type=int, help="Index of this part (PBS_ARRAYID)")
    parser.add_argument("n_parts", type=int, help="Total number of parts (nodes)")
    args = parser.parse_args(argv)

    if args.part_id < 0 or args.part_id >= args.n_parts:
        raise SystemExit(f"part_id must be in [0, {args.n_parts-1}]")

    compute_A_eta, build_eta_grid = _load_module()

    # Build full alpha grid and split into parts
    _, _, alpha_full = build_eta_grid()
    total = len(alpha_full)
    if args.n_parts <= 0:
        raise SystemExit("n_parts must be positive")

    chunk = total // args.n_parts
    # Ensure at least one element for the last chunk if not divisible
    start = args.part_id * chunk
    end = (args.part_id + 1) * chunk if args.part_id < args.n_parts - 1 else total

    # Handle edge case where chunk could be zero (more parts than grid points)
    if chunk == 0:
        # Distribute the first `total` parts one element each
        start = min(args.part_id, total)
        end = min(args.part_id + 1, total)

    subgrid = alpha_full[start:end]

    if subgrid.size == 0:
        # Nothing to do for this part
        print(f"Part {args.part_id}: empty subgrid; skipping.")
        return 0

    # Run computation on this node using 72 processes
    out = compute_A_eta(
        alpha_grid_override=subgrid,
        output_suffix=f"_part{args.part_id}",
        n_jobs=72,
    )
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

