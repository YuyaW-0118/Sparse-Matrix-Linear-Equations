#ifndef SPARSE_MATRIX_LINEAR_EQUATIONS_CPU_MULTICG_HPP
#define SPARSE_MATRIX_LINEAR_EQUATIONS_CPU_MULTICG_HPP

#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <omp.h>
#include <mkl.h>

#include "../../sparse_matrix.h"

template <typename T>
struct MklLapackTraits;

template <>
struct MklLapackTraits<double>
{
	static lapack_int gels(int matrix_layout, char trans, lapack_int m, lapack_int n,
						   lapack_int nrhs, double *a, lapack_int lda, double *b, lapack_int ldb)
	{
		return LAPACKE_dgels(matrix_layout, trans, m, n, nrhs, a, lda, b, ldb);
	}
};

template <>
struct MklLapackTraits<float>
{
	static lapack_int gels(int matrix_layout, char trans, lapack_int m, lapack_int n,
						   lapack_int nrhs, float *a, lapack_int lda, float *b, lapack_int ldb)
	{
		return LAPACKE_sgels(matrix_layout, trans, m, n, nrhs, a, lda, b, ldb);
	}
};

/**
 * SPAI (Sparse Approximate Inverse) Preconditioner
 * Static Pattern: パターンは入力行列Aと同じと仮定します (S_M = S_A)
 */
template <typename ValueT, typename OffsetT>
inline bool SparseApproximateInversion(
	const CsrMatrix<ValueT, OffsetT> &a,
	CsrMatrix<ValueT, OffsetT> &l)
{
	// 1. 出力行列 l の初期化 (Aと同じスパースパターンを持つように設定)
	l.num_rows = a.num_rows;
	l.num_cols = a.num_cols;
	l.num_nonzeros = a.num_nonzeros;

	// メモリ確保 (MKL/NUMA対応は省略し、標準newを使用。環境に合わせて変更可)
#ifdef CUB_MKL
	l.row_offsets = (OffsetT *)mkl_malloc(sizeof(OffsetT) * (l.num_rows + 1), 64);
	l.column_indices = (OffsetT *)mkl_malloc(sizeof(OffsetT) * l.num_nonzeros, 64);
	l.values = (ValueT *)mkl_malloc(sizeof(ValueT) * l.num_nonzeros, 64);
#else
	l.row_offsets = new OffsetT[l.num_rows + 1];
	l.column_indices = new OffsetT[l.num_nonzeros];
	l.values = new ValueT[l.num_nonzeros];
#endif

// パターン(row_offsets, column_indices)をコピー
// ※ 並列コピー可能
#pragma omp parallel for
	for (OffsetT i = 0; i <= l.num_rows; ++i)
		l.row_offsets[i] = a.row_offsets[i];

#pragma omp parallel for
	for (OffsetT i = 0; i < l.num_nonzeros; ++i)
		l.column_indices[i] = a.column_indices[i];

	// 2. 高速な列アクセスのために CSC形式の構造 + CSRへのマッピングを作成
	//    SPAIは列ごとに計算するため、Aの列アクセスが必要です。
	//    map[csc_idx] = csr_idx とすることで、計算結果を直接 l.values (CSR) に書き込めます。

	std::vector<OffsetT> csc_col_offsets(a.num_cols + 1, 0);
	std::vector<OffsetT> csc_row_indices(a.num_nonzeros);
	std::vector<ValueT> csc_values(a.num_nonzeros);
	std::vector<OffsetT> csc_to_csr_map(a.num_nonzeros); // マッピング配列

	// CSC変換の前処理: 各列の要素数をカウント
	for (OffsetT i = 0; i < a.num_nonzeros; ++i)
	{
		csc_col_offsets[a.column_indices[i] + 1]++;
	}
	// 累積和 (Prefix Sum)
	for (OffsetT i = 0; i < a.num_cols; ++i)
	{
		csc_col_offsets[i + 1] += csc_col_offsets[i];
	}

	// データの詰め込み (一時的なワークスペースを利用)
	{
		std::vector<OffsetT> current_col_pos = csc_col_offsets;
		for (OffsetT r = 0; r < a.num_rows; ++r)
		{
			for (OffsetT i = a.row_offsets[r]; i < a.row_offsets[r + 1]; ++i)
			{
				OffsetT c = a.column_indices[i];
				OffsetT dest = current_col_pos[c]++;

				csc_row_indices[dest] = r;
				csc_values[dest] = a.values[i];
				csc_to_csr_map[dest] = i; // ここでCSRのインデックスを保存
			}
		}
	}

	// 3. SPAI メインループ (列ごとの並列計算)
	//    minimize || A * m_k - e_k ||_2

#pragma omp parallel
	{
		// スレッドローカルな作業領域
		std::vector<ValueT> dense_matrix;
		std::vector<ValueT> rhs;
		std::vector<OffsetT> relevant_rows;
		std::vector<OffsetT> dense_row_map; // Global Row ID -> Local Dense Row ID

		// dense_row_mapのサイズ確保 (最大行数分、初期値-1)
		// sparse性が高い場合、std::mapの方がメモリ効率が良い場合がありますが、
		// 速度優先でvectorを使用する場合はサイズに注意。ここでは安全のためmapライクな挙動をvector+cleanで実装。
		std::vector<OffsetT> global_to_local(a.num_rows, -1);

#pragma omp for schedule(static)
		for (OffsetT k = 0; k < a.num_cols; ++k)
		{
			// ステップA: 未知数パラメータ J の決定
			// Static Pattern なので、Aの第k列の非ゼロ要素の位置が、Mの第k列の非ゼロ位置 (J)
			OffsetT j_start = csc_col_offsets[k];
			OffsetT j_end = csc_col_offsets[k + 1];
			OffsetT num_vars = j_end - j_start; // 未知数の数 (|J|)

			if (num_vars == 0)
				continue;

			// ステップB: 制約式に関与する行 I の特定
			// Jに含まれる列が非ゼロを持つ行の集合和
			relevant_rows.clear();
			for (OffsetT idx = j_start; idx < j_end; ++idx)
			{
				// Mのk列目の非ゼロ要素の行インデックス (row of M) = Aの列インデックス (col of A)
				// ※ Static Pattern (S_M = S_A) のため、M(j, k) != 0 となる j は、
				//    CSC化したAのk列目の非ゼロ要素の行インデックス群と一致します。
				OffsetT col_idx_in_a = csc_row_indices[idx];

				// Aの col_idx_in_a 列目の非ゼロ要素を持つ行を探す
				OffsetT a_col_start = csc_col_offsets[col_idx_in_a];
				OffsetT a_col_end = csc_col_offsets[col_idx_in_a + 1];

				for (OffsetT a_idx = a_col_start; a_idx < a_col_end; ++a_idx)
				{
					OffsetT row_idx_in_a = csc_row_indices[a_idx];
					if (global_to_local[row_idx_in_a] == -1)
					{
						global_to_local[row_idx_in_a] = (OffsetT)relevant_rows.size();
						relevant_rows.push_back(row_idx_in_a);
					}
				}
			}

			OffsetT num_equations = (OffsetT)relevant_rows.size(); // 式の数 (|I|)

			// ステップC: 密行列 (Least Squares Problem) の構築
			// Size: num_equations x num_vars
			// A_hat * x = e_k_hat
			dense_matrix.assign(num_equations * num_vars, 0.0);
			rhs.assign(num_equations, 0.0);

			// 右辺ベクトル e_k の設定 (行 k が relevant_rows に含まれていれば 1)
			if (global_to_local[k] != -1)
			{
				rhs[global_to_local[k]] = 1.0;
			}

			// 係数行列 A_hat の充填
			// J (未知数に対応する列) をループ
			for (OffsetT j_local = 0; j_local < num_vars; ++j_local)
			{
				// CSC上のインデックス
				OffsetT original_csc_idx = j_start + j_local;
				OffsetT col_idx_in_a = csc_row_indices[original_csc_idx]; // これがAにおける実際の列番号

				// Aのこの列の非ゼロ要素を走査
				for (OffsetT a_idx = csc_col_offsets[col_idx_in_a]; a_idx < csc_col_offsets[col_idx_in_a + 1]; ++a_idx)
				{
					OffsetT row = csc_row_indices[a_idx];
					OffsetT row_local = global_to_local[row];

					// row_local は必ず -1 でないはず (ステップBで収集済みのため)
					// LAPACKE_dgels はデフォルトで Column Major を好むが、
					// LAPACK_ROW_MAJOR 指定で Row Major (C言語配列) として渡す。
					// dense_matrix[row * cols + col]
					dense_matrix[row_local * num_vars + j_local] = csc_values[a_idx];
				}
			}

			// ステップD: QR分解で最小二乗問題を解く (MKL)
			// min || A_hat * x - rhs ||
			lapack_int info = MklLapackTraits<ValueT>::gels(
				LAPACK_ROW_MAJOR,
				'N',
				num_equations,
				num_vars,
				1,
				dense_matrix.data(),
				num_vars, // LDA
				rhs.data(),
				1 // LDB
			);

			// ステップE: 解の書き戻し
			// rhsの先頭 num_vars 個に解 x が入っている
			if (info == 0)
			{
				for (OffsetT j_local = 0; j_local < num_vars; ++j_local)
				{
					ValueT val = rhs[j_local];

					// 書き込み先:
					// A_cscの列kの j_local 番目の要素に対応する CSR の位置
					OffsetT original_csc_idx = j_start + j_local;
					OffsetT target_csr_idx = csc_to_csr_map[original_csc_idx];

					l.values[target_csr_idx] = val;
				}
			}
			else
			{
				// 収束失敗時のフォールバック (0埋め等)
				for (OffsetT j_local = 0; j_local < num_vars; ++j_local)
				{
					OffsetT original_csc_idx = j_start + j_local;
					l.values[csc_to_csr_map[original_csc_idx]] = 0.0;
				}
			}

			// Clean up global_to_local map for next iteration
			for (OffsetT r : relevant_rows)
			{
				global_to_local[r] = -1;
			}
		}
	}

	return true;
}

#endif // SPARSE_MATRIX_LINEAR_EQUATIONS_CPU_MULTICG_HPP
