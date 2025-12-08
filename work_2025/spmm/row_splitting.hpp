#ifndef ROW_SPLITTING_HPP
#define ROW_SPLITTING_HPP

#include <omp.h>
#include <vector>

#include <sparse_matrix.h>
#include <work_2025/types.hpp>
#include <work_2025/hyper_parameters.hpp>

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
	int num_vectors,
	ValueT *__restrict vector_x_row_major)
{
	int num_cols = a.num_cols;
	int num_rows = a.num_rows;

	if (!g_input_row_major)
	{
#pragma omp parallel for schedule(static) num_threads(num_threads)
		for (int i = 0; i < num_cols; i++)
			for (int j = 0; j < num_vectors; j++)
				vector_x_row_major[i * num_vectors + j] = vector_x[j * num_cols + i];
	}

#pragma omp parallel for schedule(static) num_threads(num_threads)
	for (OffsetT row = 0; row < a.num_rows; ++row)
	{
		std::vector<ValueT> partial(num_vectors, 0.0);
		for (
			OffsetT offset = a.row_offsets[row];
			offset < a.row_offsets[row + 1];
			++offset)
		{
			ValueT val = a.values[offset];
			int ind = a.column_indices[offset] * num_vectors;
			for (int i = 0; i < num_vectors; i++)
			{
				partial[i] += val * vector_x_row_major[ind + i];
			}
		}

		if (g_output_row_major)
		{
			int ind = row * num_vectors;
			for (int i = 0; i < num_vectors; i++)
			{
				vector_y_out[ind + i] = partial[i];
			}
		}
		else
		{
			for (int i = 0; i < num_vectors; i++)
			{
				vector_y_out[row + i * num_rows] = partial[i];
			}
		}
	}
}

#endif // ROW_SPLITTING_HPP
