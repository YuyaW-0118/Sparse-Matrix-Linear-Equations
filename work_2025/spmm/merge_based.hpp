#ifndef MERGE_BASED_HPP
#define MERGE_BASED_HPP

#include <algorithm>
#include <vector>

#include <omp.h>

#include "../../sparse_matrix.h"
#include "../types.hpp"
#include "../hyper_parameters.hpp"
#include "utils.hpp"

/**
 * Computes the begin offsets into A and B for the specific diagonal
 */
template <
	typename AIteratorT,
	typename BIteratorT,
	typename OffsetT,
	typename CoordinateT>
inline void MergePathSearch(
	OffsetT diagonal,			  ///< [in]The diagonal to search
	AIteratorT a,				  ///< [in]List A
	BIteratorT b,				  ///< [in]List B
	OffsetT a_len,				  ///< [in]Length of A
	OffsetT b_len,				  ///< [in]Length of B
	CoordinateT &path_coordinate) ///< [out] (x,y) coordinate where diagonal intersects the merge path
{
	OffsetT x_min = std::max(diagonal - b_len, 0);
	OffsetT x_max = std::min(diagonal, a_len);

	while (x_min < x_max)
	{
		OffsetT x_pivot = (x_min + x_max) >> 1;
		if (a[x_pivot] <= b[diagonal - x_pivot - 1])
			x_min = x_pivot + 1; // Contract range up A (down B)
		else
			x_max = x_pivot; // Contract range down A (up B)
	}

	path_coordinate.x = std::min(x_min, a_len);
	path_coordinate.y = diagonal - x_min;
}

template <
	typename ValueT,
	typename OffsetT>
inline void OmpMergeCsrmm(
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
				vector_x_row_major[i * num_vectors + j] = vector_x[i + (long long)j * num_rows];
	}

#pragma omp parallel for schedule(static) num_threads(num_threads)
	for (int tid = 0; tid < num_threads; tid++)
	{
		// Merge list B (NZ indices)
		CountingInputIterator<OffsetT> nonzero_indices(0);

		OffsetT num_merge_items = a.num_rows + a.num_nonzeros;						  // Merge path total length
		OffsetT items_per_thread = (num_merge_items + num_threads - 1) / num_threads; // Merge items per thread

		// Find starting and ending MergePath coordinates (row-idx, nonzero-idx) for each thread
		int2 thread_coord;
		int2 thread_coord_end;
		int start_diagonal = std::min(items_per_thread * tid, num_merge_items);
		int end_diagonal = std::min(start_diagonal + items_per_thread, num_merge_items);

		MergePathSearch(start_diagonal, row_end_offsets, nonzero_indices, a.num_rows, a.num_nonzeros, thread_coord);
		MergePathSearch(end_diagonal, row_end_offsets, nonzero_indices, a.num_rows, a.num_nonzeros, thread_coord_end);
		// Consume whole rows
		std::vector<ValueT> running_total(num_vectors, 0.0);
		ValueT val;
		int ind;
		ValueT *tmp;
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

			if (g_output_row_major)
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

#endif // MERGE_BASED_HPP
