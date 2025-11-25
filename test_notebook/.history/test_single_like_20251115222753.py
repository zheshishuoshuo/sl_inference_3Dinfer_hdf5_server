import numpy as np
import matplotlib.pyplot as plt
import sys

# ---------------------------------------------------------
#  添加工程路径
# ---------------------------------------------------------
sys.path.append("../..")

from sl_inference_3Dinfer_hdf5_server.mock_generator.mock_generator import run_mock_simulation
from sl_inference_3Dinfer_hdf5_server.main import build_dm_grid2d
from sl_inference_3Dinfer_hdf5_server.make_tabulate import tabulate_likelihood_grids
from sl_inference_3Dinfer_hdf5_server.likelihood import log_posterior


# =====================================================================
#  一维 posterior 扫描函数
# =====================================================================
def scan_1d_posterior(
    scan_values,
    *,
    fixed_alpha=0.0,
    fixed_mu_h=12.91,
    fixed_mu_gamma=1.0,
    scan_param="mu_h",
    grids=None,
):
    if grids is None:
        raise ValueError("You must pass the precomputed `grids`.")

    logpost = []

    for val in scan_values:
        alpha = fixed_alpha
        mu_h = fixed_mu_h
        mu_gamma = fixed_mu_gamma

        if scan_param == "alpha":
            alpha = float(val)
        elif scan_param == "mu_h":
            mu_h = float(val)
        elif scan_param == "mu_gamma":
            mu_gamma = float(val)
        else:
            raise ValueError("scan_param must be 'alpha', 'mu_h', or 'mu_gamma'.")

        eta = np.array([alpha, mu_h, mu_gamma], dtype=float)
        logpost.append(log_posterior(eta, grids))

    return np.array(logpost)


# =====================================================================
#  主逻辑
# =====================================================================
def main():

    # -----------------------------------------------------
    # Step 1: mock 数据
    # -----------------------------------------------------
    n_galaxy = 100000
    logalpha_true = 0.0
    seed = 123

    df_lens_all, mock_lens_data, mock_observed_data, samples = run_mock_simulation(
        n_galaxy,
        logalpha=logalpha_true,
        seed=seed,
        nbkg=4e-4,
        if_source=True,
        process=0,
    )

    print("mock_lens_data shape =", mock_lens_data.shape)
    print("mock_observed_data shape =", mock_observed_data.shape)

    # -----------------------------------------------------
    # Step 2: DM grid
    # -----------------------------------------------------
    dm_grid = build_dm_grid2d(
        logMh_min=11.0, logMh_max=15.0, n_logMh=50,
        gamma_min=0.4, gamma_max=1.6, n_gamma=50,
    )

    # -----------------------------------------------------
    # Step 3: tabulate lens kernel
    # ★ 用 mock_lens_data，不要用 mock_observed_data
    # -----------------------------------------------------
    grids = tabulate_likelihood_grids(
        mock_lens_data,
        dm_grid,
        zl=0.3,
        zs=2.0,
        n_jobs=1,        # ★ 脚本模式强烈建议单进程，不会翻车
        show_progress=True,
    )

    print("Number of LensGrid2D grids =", len(grids))

    # -----------------------------------------------------
    # Step 4: 扫描 μ_h
    # -----------------------------------------------------
    scan_mu_h = np.linspace(12.4, 13.4, 100)
    logpost_mu_h = scan_1d_posterior(
        scan_mu_h,
        fixed_alpha=0.0,
        fixed_mu_h=12.91,
        fixed_mu_gamma=1.0,
        scan_param="mu_h",
        grids=grids,
    )

    logpost_mu_h -= np.max(logpost_mu_h)
    post_mu_h = np.exp(logpost_mu_h)

    plt.figure(figsize=(6,4))
    plt.plot(scan_mu_h, post_mu_h)
    plt.axvline(12.91, color='r', ls='--', label='true')
    plt.xlabel(r"$\mu_h$")
    plt.ylabel("Posterior (norm.)")
    plt.title("1D posterior of mu_h")
    plt.legend()
    plt.savefig("posterior_mu_h.png", dpi=150)
    plt.close()

    # -----------------------------------------------------
    # Step 5: 扫描 α_sps
    # -----------------------------------------------------
    scan_alpha = np.linspace(-0.3, 0.3, 100)
    logpost_alpha = scan_1d_posterior(
        scan_alpha,
        fixed_alpha=0.0,
        fixed_mu_h=12.91,
        fixed_mu_gamma=1.0,
        scan_param="alpha",
        grids=grids,
    )

    logpost_alpha -= np.max(logpost_alpha)
    post_alpha = np.exp(logpost_alpha)

    plt.figure(figsize=(6,4))
    plt.plot(scan_alpha, post_alpha)
    plt.axvline(0.0, color='r', ls='--', label='true')
    plt.xlabel(r"$\alpha_{\rm sps}$")
    plt.ylabel("Posterior (norm.)")
    plt.title("1D posterior of alpha_sps")
    plt.legend()
    plt.savefig("posterior_alpha_sps.png", dpi=150)
    plt.close()

    # -----------------------------------------------------
    # Step 6: 扫描 μ_γ
    # -----------------------------------------------------
    scan_mu_gamma = np.linspace(0.5, 1.5, 100)
    logpost_mu_gamma = scan_1d_posterior(
        scan_mu_gamma,
        fixed_alpha=0.0,
        fixed_mu_h=12.91,
        fixed_mu_gamma=1.0,
        scan_param="mu_gamma",
        grids=grids,
    )

    logpost_mu_gamma -= np.max(logpost_mu_gamma)
    post_mu_gamma = np.exp(logpost_mu_gamma)

    plt.figure(figsize=(6,4))
    plt.plot(scan_mu_gamma, post_mu_gamma)
    plt.axvline(1.0, color='r', ls='--', label='true')
    plt.xlabel(r"$\mu_\gamma$")
    plt.ylabel("Posterior (norm.)")
    plt.title("1D posterior of mu_gamma")
    plt.legend()
    plt.savefig("posterior_mu_gamma.png", dpi=150)
    plt.close()

    print("All posterior plots saved.")


# =====================================================================
#  main-guard（防止 multiprocessing 自我复制炸内存）
# =====================================================================
if __name__ == "__main__":
    main()
