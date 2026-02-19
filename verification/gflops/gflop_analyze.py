import pandas as pd
import os
import glob

# データパス設定
data_dir = "../../data/gflops"

# 対象ファイルをすべて取得
file_pattern = os.path.join(data_dir, "*_gflops.csv")
csv_files = glob.glob(file_pattern)

# 全データを読み込み結合
df_list = []
for f in csv_files:
    try:
        temp_df = pd.read_csv(f)
        df_list.append(temp_df)
    except Exception as e:
        print(f"Error reading {f}: {e}")

if not df_list:
    print("No csv files found.")
    exit()

full_df = pd.concat(df_list, ignore_index=True)

# 値のフォーマットを作成: {gflops}({iterations})
full_df["formatted_value"] = (
    full_df["gflops"].astype(str) + "(" + full_df["iterations"].astype(str) + ")"
)

# カーネル名のマッピング（入力CSVの値を前回のkernel_type定義に合わせる）
kernel_mapping = {"SIMPLE": "simple", "MERGE": "merge", "NONZERO_SPLIT": "nonzero"}

# 存在するカーネル種別ごとに処理
unique_kernels = full_df["kernel"].unique()

for raw_kernel in unique_kernels:
    # カーネルでフィルタリング
    kernel_df = full_df[full_df["kernel"] == raw_kernel]

    # ピボットテーブル作成
    # 行: matrix_name, 列: num_vectors, 値: formatted_value
    pivot_df = kernel_df.pivot_table(
        index="matrix_name",
        columns="num_vectors",
        values="formatted_value",
        aggfunc="first",
    )

    # 列（num_vectors）を数値順にソート（2, 4, 8...の順にするため）
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

    # 保存ファイル名の決定
    kernel_type = kernel_mapping.get(raw_kernel, raw_kernel.lower())
    output_filename = f"{kernel_type}_gflops_data.csv"
    output_path = os.path.join(data_dir, output_filename)

    # CSV保存 (indexラベルをmtx_nameに設定)
    pivot_df.to_csv(output_path, index_label="mtx_name")
    print(f"Saved: {output_path}")
