import numpy as np
from sl_inference_3Dinfer_hdf5_server.tests_eta.likelihood_slices import (
    load_grids_from_hdf5,
    compute_1d_loglike,
)
import matplotlib.pyplot as plt
from sl_inference_3Dinfer_hdf5_server.mock_generator.mass_sampler import MODEL_PARAMS

# -----------------------------
# 1. 读取 grids 文件
# -----------------------------
grids_path = "./data/mock_3/grids_20251119T210801Z.h5"  # <-- 替换成你的 grids 文件路径
grids = load_grids_from_hdf5(grids_path)

# -----------------------------
# 2. 设置 alpha_sps 网格
# -----------------------------
alpha_true = 0.0   # ground truth
alpha_grid = np.linspace(-0.025, 0.025, 200)  # 高分辨率网格

# -----------------------------
# 3. 计算一维后验（log-likelihood）
# -----------------------------
logL_alpha = compute_1d_loglike(
    grids,
    param_name="alpha_sps",
    param_grid=alpha_grid,
    fixed_params=None,       # 不固定其它参数
    use_prior=True,
    use_A_eta=True,
)

# -----------------------------
# 4. 后验归一化
# -----------------------------
# posterior ∝ exp(logL)
posterior = np.exp(logL_alpha - np.max(logL_alpha))  # 防止溢出
posterior /= np.trapz(posterior, alpha_grid)         # 归一化成 PDF

# -----------------------------
# 5. 绘图
# -----------------------------
plt.figure(figsize=(7,5))
plt.plot(alpha_grid, posterior, lw=2)
plt.axvline(alpha_true, color="red", linestyle="--", label="True α_sps")
plt.xlabel("alpha_sps")
plt.ylabel("Posterior P(alpha_sps | data)")
plt.title("1D Posterior of alpha_sps")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
