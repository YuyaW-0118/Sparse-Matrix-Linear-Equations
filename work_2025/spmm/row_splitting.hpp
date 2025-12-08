#ifndef ROW_SPLITTING_HPP
#define ROW_SPLITTING_HPP

#include <omp.h>
#include <cstring>

#include "../../sparse_matrix.h"
#include "../types.hpp"
#include "../hyper_parameters.hpp"

//---------------------------------------------------------------------
// CPU normal omp SpMV
//---------------------------------------------------------------------

template <
	typename ValueT,
	typename OffsetT>
inline void OmpCsrSpmmT(
	int num_threads,
	CsrMatrix<ValueT, OffsetT> &a,
	ValueT *__restrict vector_x,
	ValueT *__restrict vector_y_out,
	int num_vectors)
{
	int num_cols = a.num_cols;
	int num_rows = a.num_rows;

#pragma omp parallel for schedule(static) num_threads(num_threads)
	for (OffsetT row = 0; row < a.num_rows; ++row)
	{
		ValueT row_sum[num_vectors];
		for (int i = 0; i < num_vectors; i++)
			row_sum[i] = 0.0;

		OffsetT row_end = a.row_offsets[row + 1];
		for (OffsetT offset = a.row_offsets[row]; offset < row_end; ++offset)
		{
			ValueT val = a.values[offset];
			OffsetT col = a.column_indices[offset];
			OffsetT x_col_start = col * num_vectors;
#pragma omp simd
			for (int i = 0; i < num_vectors; i++)
			{
				row_sum[i] += val * vector_x[x_col_start + i];
			}
		}
		OffsetT y_row_start = row * num_vectors;
#pragma omp simd
		for (int i = 0; i < num_vectors; i++)
		{
			vector_y_out[y_row_start + i] = row_sum[i];
		}
	}
}

#endif // ROW_SPLITTING_HPP
