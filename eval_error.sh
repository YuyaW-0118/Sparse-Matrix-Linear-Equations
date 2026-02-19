#!/bin/bash

# download/ディレクトリ以下の全.mtxファイルに対してCGソルバーを実行し、エラーをプロットする

DOWNLOAD_DIR="/home/yuya/projects/school/Sparse-Matrix-Linear-Equations/download/final_mtx2"

# 全.mtxファイルを検索
find "$DOWNLOAD_DIR" -name "*.mtx" | while read mtx_path; do
    # ファイル名から拡張子を除いた名前を取得
    mtx_name=$(basename "$mtx_path" .mtx)

    echo "Processing: $mtx_name"

    # CGソルバーを実行
    ./_cpu_multicg_driver --mtx="$mtx_path" --quiet

    # エラープロットを生成
    python verification/plot_errors.py "$mtx_name"

    echo "Completed: $mtx_name"
    echo "---"
done
