import os
import glob
import h5py
import numpy as np


def _list_part_files(cwd: str) -> list[str]:
    pattern = os.path.join(cwd, "*_part*.h5")
    files = sorted(glob.glob(pattern))
    return files


def _load_grids(path: str):
    with h5py.File(path, "r") as f:
        mu_DM = np.array(f["grids/mu_DM_grid"]) if "grids/mu_DM_grid" in f else None
        mu_gamma = np.array(f["grids/mu_gamma_grid"]) if "grids/mu_gamma_grid" in f else None
        alpha = np.array(f["grids/alpha_grid"]) if "grids/alpha_grid" in f else None
        A = np.array(f["grids/A_grid"]) if "grids/A_grid" in f else None
        meta = dict(f["metadata"].attrs) if "metadata" in f else {}
    if any(x is None for x in (mu_DM, mu_gamma, alpha, A)):
        raise ValueError(f"Missing required datasets in {path}")
    return mu_DM, mu_gamma, alpha, A, meta


def main():
    cwd = os.getcwd()
    files = _list_part_files(cwd)
    if not files:
        raise SystemExit("No _part*.h5 files found in current directory")

    # Load all parts
    parts = []
    for fp in files:
        mu_DM, mu_gamma, alpha, A, meta = _load_grids(fp)
        parts.append((fp, mu_DM, mu_gamma, alpha, A, meta))

    # Sort by the first alpha value to ensure correct order along alpha axis
    parts.sort(key=lambda x: (float(x[3][0]) if len(x[3]) > 0 else np.inf))

    # Validate mu grids are identical
    ref_mu_DM = parts[0][1]
    ref_mu_gamma = parts[0][2]
    for fp, mu_DM, mu_gamma, alpha, A, meta in parts[1:]:
        if not (np.array_equal(mu_DM, ref_mu_DM) and np.array_equal(mu_gamma, ref_mu_gamma)):
            raise SystemExit(f"mu grids mismatch in {fp}")

    # Concatenate along alpha axis (axis=2: mu_DM x mu_gamma x alpha)
    alpha_all = np.concatenate([p[3] for p in parts], axis=0)
    A_all = np.concatenate([p[4] for p in parts], axis=2)

    out_path = os.path.join(cwd, "Aeta3D_merged.h5")
    with h5py.File(out_path, "w") as f:
        # Write metadata from the first file (preserve fields)
        gmeta = f.create_group("metadata")
        for k, v in parts[0][5].items():
            gmeta.attrs[k] = v

        # Grids and merged A table
        g = f.create_group("grids")
        g.create_dataset("mu_DM_grid", data=ref_mu_DM)
        g.create_dataset("mu_gamma_grid", data=ref_mu_gamma)
        g.create_dataset("alpha_grid", data=alpha_all)
        g.create_dataset("A_grid", data=A_all.astype(np.float32), compression="gzip")

    print(f"Wrote merged file: {out_path}")


if __name__ == "__main__":
    main()

