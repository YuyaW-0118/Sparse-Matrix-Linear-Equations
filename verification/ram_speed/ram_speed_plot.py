from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def plot_bandwidth_for_threads(
    n: int, csv_path: str | Path, out_path: str | Path
) -> None:
    """
    csv_path を読み、スレッド数 n の「データサイズ[KB] vs バンド幅[GB/s]」をプロットして out_path に保存する。

    対応するCSV形式（どれかに合えばOK）:
    """
    csv_path = Path(csv_path)
    out_path = Path(out_path)

    def _to_float_row(r: Sequence[str]) -> list[float]:
        return [float(x) for x in r]

    rows: dict[float, list[float]] = {}
    with csv_path.open(newline="") as f:
        for r in csv.reader(f):
            r = [x.strip() for x in r if x.strip() != ""]
            float_row = _to_float_row(r)
            val = float_row[0]
            rows[val] = float_row[1:]

    if not rows:
        raise ValueError("CSVが空です。")

    sorted_sizes = sorted(rows.keys())
    x: list[float] = []
    y: list[float] = []

    for i, size_kb in enumerate(sorted_sizes):
        bandwidths = rows[size_kb]
        if len(bandwidths) >= n:
            bw = bandwidths[n - 1]
            x.append(size_kb)
            y.append(bw)

    # ---- plot
    fontsize = 12
    plt.figure()
    plt.plot(x, y, color="red")
    plt.xscale("log", base=2)
    plt.yscale("log", base=10)
    plt.xlim(2**2, 2**20)
    plt.ylim(10**1, 10**4)
    plt.xticks([2**i for i in range(2, 19, 2)])
    plt.yticks([10**i for i in range(1, 5)])
    plt.xlabel("Data size (KB)", fontsize=fontsize)
    plt.ylabel("Bandwidth (GB/s)", fontsize=fontsize)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # L1領域
    plt.text(
        2**3.5,
        7000,
        "L1 Cache",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="black",
    )
    # L2領域
    plt.text(
        2**7.5,
        7000,
        "L2 Cache",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="black",
    )
    # L3領域 (と推測される1MB~128MB区間)
    plt.text(
        2**12.5,
        3000,
        "L3 Cache",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="black",
    )
    # RAM領域 (128MB以降)
    plt.text(
        2**18,
        100,
        "Main Memory",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="black",
    )

    # 縦線とラベル
    cache_labels = [(32, "L1(32KB)"), (1024, "L2(1024KB)"), (25344, "L3(25344KB)")]
    for vline_x, label in cache_labels:
        plt.axvline(x=vline_x, color="blue", linestyle="--", linewidth=1)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    n = 18
    csv_path = "../../data/result_Intel_Core_i9-9980XE_3.00GHz_physical.csv"
    out_path = f"../../data/plots/ram_bandwidth_{n}threads.png"

    plot_bandwidth_for_threads(n, csv_path, out_path)
