import numpy as np
import matplotlib.pyplot as plt
import sys

# ---------------------------------------------------------
#  添加工程路径（按你的工程结构，只需要加一次）
# ---------------------------------------------------------
sys.path.append("../..")

# 正确导入你的模块
from sl_inference_3Dinfer_hdf5_server.mock_generator.mock_generator import run_mock_simulation
from sl_inference_3Dinfer_hdf5_server.main import build_dm_grid2d
from sl_inference_3Dinfer_hdf5_server.make_tabulate import tabulate_likelihood_grids
from sl_inference_3Dinfer_hdf5_server.likelihood import log_posterior


# =====================================================================
#  0. 统一的“一维 posterior 扫描”函数
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
    """
    对 3D 超参数 η = (alpha_sps, mu_h, mu_gamma) 做一维 posterior 扫描。
    
    参数:
        scan_values: 要扫描的点，例如 np.linspace(...)
        scan_param: 'alpha' or 'mu_h' or 'mu_gamma'
        fixed_xxx: 其余两个保持固定
        grids: tabulate_likelihood_grids() 生成的 list(LensGrid2D)，必须传入

    返回:
        logpost: 与 scan_values 对应的 log posterior 数组（未归一化）
    """
    if grids is None:
        raise ValueError("You must pass the precomputed `grids`.")

    logpost = []

    for val in scan_values:
        alpha = fixed_alpha
        mu_h = fixed_mu_h
        mu_gamma = fixed_mu_gamma

        # 替换扫描变量
        if scan_param == "alpha":
            alpha = float(val)
        elif scan_param == "mu_h":
            mu_h = float(val)
        elif scan_param == "mu_gamma":
            mu_gamma = float(val)
        else:
            raise ValueError("scan_param must be one of: 'alpha', 'mu_h', 'mu_gamma'.")

        eta = np.array([alpha, mu_h, mu_gamma], dtype=float)

        # log_posterior 内部会自动:
        #  - 对每个 lens 调用 single_lens_likelihood
        #  - 使用 tabulated A_interp 做 selection correction
        lp = log_posterior(eta, grids)
        logpost.append(lp)

    return np.array(logpost)


# =====================================================================
#  1. Step 1 —— 生成 mock 数据（与 main.py 完全一致）
# =====================================================================
n_galaxy = 100000        # 小样本用于测试
logalpha_true = 0.0  # mock 的真实 alpha_sps
seed = 123

df_lens_all, mock_lens_data, mock_observed_data, samples = run_mock_simulation(
    n_galaxy,
    logalpha=logalpha_true,
    seed=seed,
    nbkg=4e-4,
    if_source=True,
    process=10,
)

print("Number of lensed systems:", mock_lens_data.shape[0])


# =====================================================================
#  2. Step 2 —— 构建 (logMh, gamma) DM grid
# =====================================================================
dm_grid = build_dm_grid2d(
    logMh_min=11.0, logMh_max=15.0, n_logMh=50,
    gamma_min=0.4, gamma_max=1.6, n_gamma=50,
)


# =====================================================================
#  3. Step 3 —— 为每个 lens tabulate 2D kernel（核心步骤）
# =====================================================================
grids = tabulate_likelihood_grids(
    mock_observed_data,
    dm_grid,
    zl=0.3,
    zs=2.0,
    n_jobs=10,        # 为了 notebook 调试，先别开并行
    show_progress=True,
)

print("Number of LensGrid2D grids = ", len(grids))


# =====================================================================
#  4. Step 4 —— 三种参数的 1D posterior 示例
# =====================================================================

# ---------------------------------------------------------------------
# 扫描 μ_h
# ---------------------------------------------------------------------
scan_mu_h = np.linspace(12.4, 13.4, 100)

logpost_mu_h = scan_1d_posterior(
    scan_mu_h,
    fixed_alpha=0.0,
    fixed_mu_h=12.91,   # 只是占位，扫描时会被覆盖
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
plt.show()


# ---------------------------------------------------------------------
# 扫描 alpha_sps
# ---------------------------------------------------------------------
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
plt.show()


# ---------------------------------------------------------------------
# 扫描 μ_gamma
# ---------------------------------------------------------------------
scan_mu_gamma = np.linspace(0.5, 1.5, 100)

logpost_mu_gamma = scan_1d_posterior(
    scan_mu_gamma,
    fixed_alpha=0.0,
    fixed_mu_h=12.91,
    fixed_mu_gamma=1.0,   # 只是占位符
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
plt.show()
