import pandas as pd
import os
import glob

# データパス設定
base_dir = "/home/yuya/projects/school/Sparse-Matrix-Linear-Equations/data/prepare"
output_path = os.path.join(base_dir, "preconditioner_summary.csv")

# 対象ファイルをすべて取得
file_pattern = os.path.join(base_dir, "*_prepare.csv")
csv_files = glob.glob(file_pattern)

summary_list = []

for f in csv_files:
    try:
        # ファイル名から行列名を取得
        mtx_name = os.path.basename(f).replace("_prepare.csv", "")

        # CSV読み込み (1列目をインデックスとして扱う)
        df = pd.read_csv(f, index_col=0, skipinitialspace=True)

        # 必要なメソッドが存在するか確認
        required_methods = ["NONE", "IC0", "SPAI"]
        if not all(m in df.index for m in required_methods):
            print(f"Skipping {mtx_name}: Missing required methods.")
            continue

        # 1行分のデータを作成
        row_data = {"mtx_name": mtx_name}

        # 基本データの抽出 (NONEの時間を基準にするため取得)
        time_none = df.loc["NONE", "total_ms"]

        for method in required_methods:
            # 各メソッドの生データ
            row_data[f"preprocess_ms_{method}"] = df.loc[method, "preprocess_ms"]
            row_data[f"solve_ms_{method}"] = df.loc[method, "solve_ms"]
            row_data[f"total_ms_{method}"] = df.loc[method, "total_ms"]
            row_data[f"iter_{method}"] = df.loc[method, "iterations"]
            row_data[f"gflops_{method}"] = df.loc[method, "gflops"]

            # Speedupの計算 (対 NONE)
            if time_none > 0 and df.loc[method, "total_ms"] > 0:
                speedup = time_none / df.loc[method, "total_ms"]
            else:
                speedup = 0.0

            row_data[f"speedup_{method}"] = round(speedup, 4)

        summary_list.append(row_data)

    except Exception as e:
        print(f"Error processing {f}: {e}")

if not summary_list:
    print("No valid data found.")
    exit()

# 修正箇所: リストから直接DataFrameを作成
result_df = pd.DataFrame(summary_list)

# カラムの並び順を整理
cols = ["mtx_name"]
metrics = ["total_ms", "speedup", "iter", "preprocess_ms", "solve_ms", "gflops"]
methods = ["NONE", "IC0", "SPAI"]

ordered_cols = ["mtx_name"]
for metric in metrics:
    for method in methods:
        col_name = f"{metric}_{method}"
        if col_name in result_df.columns:
            ordered_cols.append(col_name)

# 並び替えて保存
result_df = result_df[ordered_cols]
result_df.to_csv(output_path, index=False)

print(f"Summary saved to: {output_path}")
