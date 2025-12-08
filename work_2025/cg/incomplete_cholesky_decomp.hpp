#ifndef INCOMPLETE_CHOLESKY_DECOMP_HPP
#define INCOMPLETE_CHOLESKY_DECOMP_HPP

#include <mkl.h>
#include <omp.h>

#include "../../sparse_matrix.h"

/**
 * Transposes a CSR matrix.
 */
template <typename ValueT, typename OffsetT>
inline void TransposeCsr(
	const CsrMatrix<ValueT, OffsetT> &in,
	CsrMatrix<ValueT, OffsetT> &out)
{
	out.num_rows = in.num_cols;
	out.num_cols = in.num_rows;
	out.num_nonzeros = in.num_nonzeros;

	out.row_offsets = new OffsetT[out.num_rows + 1];
	out.column_indices = new OffsetT[out.num_nonzeros];
	out.values = new ValueT[out.num_nonzeros];

	OffsetT *col_counts = new OffsetT[out.num_rows];
	std::fill(col_counts, col_counts + out.num_rows, 0);

	for (OffsetT i = 0; i < in.num_nonzeros; ++i)
	{
		col_counts[in.column_indices[i]]++;
	}

	OffsetT sum = 0;
	for (OffsetT i = 0; i < out.num_rows; ++i)
	{
		OffsetT temp = col_counts[i];
		out.row_offsets[i] = sum;
		sum += temp;
		col_counts[i] = out.row_offsets[i];
	}
	out.row_offsets[out.num_rows] = sum;

	for (OffsetT i = 0; i < in.num_rows; ++i)
	{
		for (OffsetT j_offset = in.row_offsets[i]; j_offset < in.row_offsets[i + 1]; ++j_offset)
		{
			OffsetT j = in.column_indices[j_offset];
			ValueT val = in.values[j_offset];

			OffsetT out_idx = col_counts[j];
			out.column_indices[out_idx] = i;
			out.values[out_idx] = val;

			col_counts[j]++;
		}
	}

	delete[] col_counts;
}

/**
 * Performs Incomplete Cholesky factorization (IC(0)) on a CSR matrix.
 * NOTE: This is a sequential implementation. The input matrix A must be symmetric.
 */
template <typename ValueT, typename OffsetT>
inline bool IncompleteCholesky(
	const CsrMatrix<ValueT, OffsetT> &a,
	CsrMatrix<ValueT, OffsetT> &l)
{
	l.num_rows = a.num_rows;
	l.num_cols = a.num_cols;

	// The sparsity pattern of L is the lower triangle of A
	std::vector<std::vector<std::pair<OffsetT, ValueT>>> l_temp(l.num_rows);
	for (OffsetT i = 0; i < a.num_rows; ++i)
	{
		for (OffsetT j_offset = a.row_offsets[i]; j_offset < a.row_offsets[i + 1]; ++j_offset)
		{
			if (a.column_indices[j_offset] <= i)
			{
				l_temp[i].push_back({a.column_indices[j_offset], a.values[j_offset]});
			}
		}
	}

	l.num_nonzeros = 0;
	for (const auto &row : l_temp)
		l.num_nonzeros += row.size();

	l.row_offsets = (OffsetT *)mkl_malloc(sizeof(OffsetT) * (l.num_rows + 1), 4096);
	l.column_indices = (OffsetT *)mkl_malloc(sizeof(OffsetT) * l.num_nonzeros, 4096);
	l.values = (ValueT *)mkl_malloc(sizeof(ValueT) * l.num_nonzeros, 4096);

	l.row_offsets[0] = 0;
	OffsetT nz_count = 0;
	for (OffsetT i = 0; i < l.num_rows; ++i)
	{
		for (const auto &p : l_temp[i])
		{
			l.column_indices[nz_count] = p.first;
			l.values[nz_count] = p.second;
			nz_count++;
		}
		l.row_offsets[i + 1] = nz_count;
	}

	for (OffsetT i = 0; i < a.num_rows; ++i)
	{
		for (OffsetT k_offset = l.row_offsets[i]; k_offset < l.row_offsets[i + 1]; ++k_offset)
		{
			OffsetT k = l.column_indices[k_offset];

			ValueT sum = 0.0;
			OffsetT j_offset_l = l.row_offsets[i];
			OffsetT j_offset_k = l.row_offsets[k];

			while (j_offset_l < k_offset && j_offset_k < l.row_offsets[k + 1])
			{
				if (l.column_indices[j_offset_l] == l.column_indices[j_offset_k])
				{
					sum += l.values[j_offset_l] * l.values[j_offset_k];
					j_offset_l++;
					j_offset_k++;
				}
				else if (l.column_indices[j_offset_l] < l.column_indices[j_offset_k])
				{
					j_offset_l++;
				}
				else
				{
					j_offset_k++;
				}
			}

			l.values[k_offset] -= sum;

			if (k == i)
			{ // Diagonal element
				if (l.values[k_offset] <= 0)
				{
					fprintf(stderr, "Error: Incomplete Cholesky failed. Not positive definite or numerically unstable.\n");
					return false;
				}
				l.values[k_offset] = sqrt(l.values[k_offset]);
			}
			else
			{													  // Off-diagonal element
				OffsetT diag_k_offset = l.row_offsets[k + 1] - 1; // Assuming diagonal is last
				l.values[k_offset] /= l.values[diag_k_offset];
			}
		}
	}
	return true;
}

/**
 * Multiple-RHS Forward substitution: Solves LX = B
 * L must be a lower triangular matrix.
 */
template <typename ValueT, typename OffsetT>
inline void ForwardSolveMultiple(
	const CsrMatrix<ValueT, OffsetT> &l,
	const ValueT *b,
	ValueT *x,
	int num_vectors)
{
	OffsetT n = l.num_rows;

	for (OffsetT i = 0; i < n; ++i)
	{
		ValueT sum[num_vectors];
#pragma omp simd
		for (int v = 0; v < num_vectors; ++v)
			sum[v] = 0.0;

		OffsetT diag_offset = 0;
		for (OffsetT j_offset = l.row_offsets[i]; j_offset < l.row_offsets[i + 1]; ++j_offset)
		{
			OffsetT j = l.column_indices[j_offset];
			ValueT val = l.values[j_offset];

			if (i == j)
			{
				diag_offset = j_offset;
				continue;
			}
			OffsetT x_srx_idx = j * num_vectors;
#pragma omp simd
			for (int v = 0; v < num_vectors; ++v)
				sum[v] += val * x[x_srx_idx + v];
		}

		ValueT diag_val = l.values[diag_offset];
		OffsetT b_srx_idx = i * num_vectors;
#pragma omp simd
		for (int v = 0; v < num_vectors; ++v)
		{
			x[b_srx_idx + v] = (b[b_srx_idx + v] - sum[v]) / diag_val;
		}
	}
}

/**
 * Multiple-RHS Backward substitution: Solves L^T * X = B
 * L is a lower triangular matrix.
 */
template <typename ValueT, typename OffsetT>
inline void BackwardSolveMultiple(
	const CsrMatrix<ValueT, OffsetT> &l_t,
	const ValueT *b,
	ValueT *x,
	int num_vectors)
{
	OffsetT n = l_t.num_rows;

	for (OffsetT i = n - 1; i >= 0; --i)
	{
		ValueT sum[num_vectors];
#pragma omp simd
		for (int v = 0; v < num_vectors; ++v)
			sum[v] = 0.0;

		ValueT diag_val = 0.0;

		for (OffsetT j_offset = l_t.row_offsets[i]; j_offset < l_t.row_offsets[i + 1]; ++j_offset)
		{
			OffsetT j = l_t.column_indices[j_offset];
			ValueT val = l_t.values[j_offset];

			if (i == j)
			{
				diag_val = val;
				continue;
			}
			OffsetT x_src_idx = j * num_vectors;

#pragma omp simd
			for (int v = 0; v < num_vectors; ++v)
			{
				sum[v] += val * x[x_src_idx + v];
			}
		}

		OffsetT idx = i * num_vectors;

		if (diag_val == 0.0)
		{
#pragma omp simd
			for (int v = 0; v < num_vectors; ++v)
				x[idx + v] = 0.0;
		}
		else
		{
#pragma omp simd
			for (int v = 0; v < num_vectors; ++v)
			{
				x[idx + v] = (b[idx + v] - sum[v]) / diag_val;
			}
		}
	}
}

#endif // INCOMPLETE_CHOLESKY_DECOMP_HPP
