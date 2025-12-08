#ifndef NONZERO_SPLITTING_HPP
#define NONZERO_SPLITTING_HPP

#include <algorithm>
#include <vector>

#include <omp.h>

#include "../../sparse_matrix.h"
#include "../types.hpp"
#include "../hyper_parameters.hpp"
#include "utils.hpp"

template <
	typename AIteratorT,
	typename BIteratorT,
	typename OffsetT,
	typename CoordinateT>
inline void RowPathSearch(
	AIteratorT a,				  ///< [in]List A
	BIteratorT b,				  ///< [in]List B
	OffsetT a_len,				  ///< [in]Length of A
	CoordinateT &path_coordinate) ///< [out] (x,y) coordinate where diagonal intersects the merge path
{
	if (path_coordinate.y == 0)
	{
		path_coordinate.x = 0;
		return;
	}

	OffsetT x_min = 0;
	OffsetT x_max = a_len;

	while (x_min < x_max)
	{
		OffsetT x_pivot = (x_min + x_max) >> 1;
		if (a[x_pivot] <= b[path_coordinate.y - 1])
			x_min = x_pivot + 1; // Contract range up A (down B)
		else
			x_max = x_pivot; // Contract range down A (up B)
	}

	path_coordinate.x = std::min(x_min, a_len);
}

/**
 * OpenMP CPU row-based SpMM
 */
template <
	typename ValueT,
	typename OffsetT>
inline void OmpNonzeroSplitCsrmm(
	int num_threads,
	CsrMatrix<ValueT, OffsetT> &a,
	OffsetT *__restrict row_end_offsets, ///< Merge list A (row end-offsets)
	OffsetT *__restrict column_indices,
	ValueT *__restrict values,
	ValueT *__restrict vector_x,
	ValueT *__restrict vector_y_out,
	int num_vectors,
	ValueT *__restrict vector_x_row_major)
{
	// Temporary storage for inter-thread fix-up after load-balanced work
	OffsetT *row_carry_out = new OffsetT[num_threads];							// The last row-id each worked on by each thread when it finished its path segment
	ValueT *value_carry_out = new ValueT[(long long)num_threads * num_vectors]; // The running total within each thread when it finished its path segment

	int num_cols = a.num_cols;
	int num_rows = a.num_rows;

	if (!g_input_row_major)
	{
#pragma omp parallel for schedule(static) num_threads(num_threads)
		for (int i = 0; i < num_rows; i++)
			for (int j = 0; j < num_vectors; j++)
				vector_x_row_major[i * num_vectors + j] = vector_x[j * num_rows + i];
	}

#pragma omp parallel for schedule(static) num_threads(num_threads)
	for (int tid = 0; tid < num_threads; tid++)
	{
		// Merge list B (NZ indices)
		CountingInputIterator<OffsetT> nonzero_indices(0);

		OffsetT num_nonzeros = a.num_nonzeros;
		OffsetT items_per_thread = (num_nonzeros + num_threads - 1) / num_threads;

		// Find starting and ending MergePath coordinates (row-idx, nonzero-idx) for each thread
		int2 thread_coord;
		int2 thread_coord_end;
		thread_coord.y = std::min(items_per_thread * tid, num_nonzeros);
		thread_coord_end.y = std::min(thread_coord.y + items_per_thread, num_nonzeros);

		RowPathSearch(row_end_offsets, nonzero_indices, a.num_rows, thread_coord);
		RowPathSearch(row_end_offsets, nonzero_indices, a.num_rows, thread_coord_end);

		// Consume whole rows
		std::vector<ValueT> running_total(num_vectors, 0.0);
		ValueT val;
		ValueT *tmp;
		int ind;
		for (; thread_coord.x < thread_coord_end.x; ++thread_coord.x)
		{
			for (; thread_coord.y < row_end_offsets[thread_coord.x]; ++thread_coord.y)
			{
				val = values[thread_coord.y];
				ind = column_indices[thread_coord.y] * num_vectors;
				tmp = vector_x_row_major + ind;
				for (int i = 0; i < num_vectors; i++)
				{
					running_total[i] += val * tmp[i];
				}
			}

			if (g_output_row_major) // g_output_row_major が利用可能であると仮定
			{
				ind = thread_coord.x * num_vectors;
				tmp = vector_y_out + ind;
				for (int i = 0; i < num_vectors; i++)
				{
					tmp[i] = running_total[i];
					running_total[i] = 0.0;
				}
			}
			else
			{
				OffsetT row_idx = thread_coord.x;
				for (int i = 0; i < num_vectors; i++)
				{
					// Column-Major で書き込み
					vector_y_out[row_idx + (long long)i * num_rows] = running_total[i];
					running_total[i] = 0.0;
				}
			}
		}

		// Consume partial portion of thread's last row
		for (; thread_coord.y < thread_coord_end.y; ++thread_coord.y)
		{
			val = values[thread_coord.y];
			ind = column_indices[thread_coord.y] * num_vectors;
			tmp = vector_x_row_major + ind;
			for (int i = 0; i < num_vectors; i++)
			{
				running_total[i] += val * tmp[i];
			}
		}

		// Save carry-outs
		row_carry_out[tid] = thread_coord_end.x;

		ValueT *my_carry_out_dest = value_carry_out + (long long)tid * num_vectors;
		std::copy(running_total.begin(), running_total.end(), my_carry_out_dest);
	}

	// Carry-out fix-up (rows spanning multiple threads)
	for (int tid = 0; tid < num_threads; ++tid)
	{
		OffsetT row_idx = row_carry_out[tid];
		if (row_idx < a.num_rows)
		{
			ValueT *my_carry_out_src = value_carry_out + (long long)tid * num_vectors;
			for (int i = 0; i < num_vectors; i++)
				vector_y_out[row_idx + (long long)i * num_rows] += my_carry_out_src[i];
		}
	}

	delete[] value_carry_out;
	delete[] row_carry_out;
}

#endif // NONZERO_SPLITTING_HPP
