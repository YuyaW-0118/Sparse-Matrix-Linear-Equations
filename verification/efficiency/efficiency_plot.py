import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# データパスと設定
base_path = "../..//data/parallel"
kernel_types = ["simple", "merge", "nonzero"]

# プロット用設定（ラベル、色、マーカー）
config = {
    "simple": {"label": "row-splitting", "color": "blue", "marker": "o"},
    "merge": {"label": "merge-path", "color": "red", "marker": "s"},
    "nonzero": {"label": "nonzero-splitting", "color": "green", "marker": "^"},
}

# 学術的なグラフスタイルの適用（フォントサイズ等）
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# データの読み込み
dfs = {}
for k_type in kernel_types:
    file_path = os.path.join(base_path, f"parallel_efficiency_{k_type}.csv")
    dfs[k_type] = pd.read_csv(file_path)

# 共通のX軸データ（スレッド数）を取得（すべてのCSVで同じと仮定）
threads = dfs["simple"]["num_threads"].values

# ==========================================
# 1. 速度向上率 (Speedup) - 折れ線グラフ
# ==========================================
plt.figure(figsize=(8, 6))

# 各カーネルのプロット
for k_type in kernel_types:
    plt.plot(
        dfs[k_type]["num_threads"],
        dfs[k_type]["speedup"],
        label=config[k_type]["label"],
        color=config[k_type]["color"],
        marker=config[k_type]["marker"],
        linestyle="-",
        linewidth=2,
        markersize=8,
    )

# Idealライン (y=x)
plt.plot(threads, threads, label="ideal", color="black", linestyle="--", linewidth=2)

max_thread = threads.max()
plt.xticks(np.arange(0, max_thread + 2, 2))
plt.xlim(0, max_thread + 1)

plt.xlabel("Number of Threads", fontsize=20)
plt.ylabel("Speedup", fontsize=20)
# plt.title("Speedup vs Number of Threads")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.tight_layout()

# 保存
output_path_speedup = os.path.join(base_path, "speedup_graph_big.png")
plt.savefig(output_path_speedup, dpi=300)
plt.close()

# ==========================================
# 2. 並列化効率 (Efficiency) - 棒グラフ
# ==========================================
plt.figure(figsize=(10, 6))

x = np.arange(len(threads))  # 棒グラフ用のX座標インデックス
width = 0.25  # 棒の幅

# 各カーネルのプロット（位置をずらして描画）
for i, k_type in enumerate(kernel_types):
    offset = (i - 1) * width  # -width, 0, +width のようにずらす
    plt.bar(
        x + offset,
        dfs[k_type]["efficiency"],
        width,
        label=config[k_type]["label"],
        color=config[k_type]["color"],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )

plt.xlabel("Number of Threads")
plt.ylabel("Efficiency")
# plt.title("Parallel Efficiency vs Number of Threads")
plt.xticks(x, threads)  # X軸のラベルをスレッド数に設定
plt.ylim(0, 1.1)  # 効率は通常1.0以下だが、凡例等のために少し余裕を持たせる
plt.legend(loc="upper right")
plt.grid(axis="y", linestyle=":", alpha=0.7)
plt.tight_layout()

# 保存
output_path_efficiency = os.path.join(base_path, "efficiency_graph.png")
plt.savefig(output_path_efficiency, dpi=300)
plt.close()

print(f"Graphs saved to:\n{output_path_speedup}\n{output_path_efficiency}")
