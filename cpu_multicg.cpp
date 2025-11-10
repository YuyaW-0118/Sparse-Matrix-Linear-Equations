/******************************************************************************
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIAeBILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * How to build:
 *
 * VC++
 *      cl.exe mergebased_spmv.cpp /fp:strict /MT /O2 /openmp
 *
 * GCC (OMP is terrible)
 *      g++ mergebased_spmv.cpp -lm -ffloat-store -O3 -fopenmp
 *
 * Intel
 *      icpc mergebased_spmv.cpp -openmp -O3 -lrt -fno-alias -xHost -lnuma
 *      export KMP_AFFINITY=granularity=core,scatter
 *
 *
 ******************************************************************************/

//---------------------------------------------------------------------
// SpMV comparison tool
//---------------------------------------------------------------------

#include <omp.h>

#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
#include <immintrin.h>

#include <mkl.h>

#include "sparse_matrix.h"
#include "utils.h"

//---------------------------------------------------------------------
// Globals, constants, and type declarations
//---------------------------------------------------------------------

bool g_quiet = false;	 // Whether to display stats in CSV format
bool g_verbose = false;	 // Whether to display output to console
bool g_verbose2 = false; // Whether to display input to console
int g_omp_threads = 10;	 // Number of openMP threads
int g_expected_calls = 1000000;
bool g_input_row_major = false;
bool g_output_row_major = false;

//---------------------------------------------------------------------
// Utility types
//---------------------------------------------------------------------

struct int2
{
	int x;
	int y;
};

/**
 * Counting iterator
 */
template <
	typename ValueType,
	typename OffsetT = ptrdiff_t>
struct CountingInputIterator
{
	// Required iterator traits
	typedef CountingInputIterator self_type;				   ///< My own type
	typedef OffsetT difference_type;						   ///< Type to express the result of subtracting one iterator from another
	typedef ValueType value_type;							   ///< The type of the element the iterator can point to
	typedef ValueType *pointer;								   ///< The type of a pointer to an element the iterator can point to
	typedef ValueType reference;							   ///< The type of a reference to an element the iterator can point to
	typedef std::random_access_iterator_tag iterator_category; ///< The iterator category

	ValueType val;

	/// Constructor
	inline CountingInputIterator(
		const ValueType &val) ///< Starting value for the iterator instance to report
		: val(val)
	{
	}

	/// Postfix increment
	inline self_type operator++(int)
	{
		self_type retval = *this;
		val++;
		return retval;
	}

	/// Prefix increment
	inline self_type operator++()
	{
		val++;
		return *this;
	}

	/// Indirection
	inline reference operator*() const
	{
		return val;
	}

	/// Addition
	template <typename Distance>
	inline self_type operator+(Distance n) const
	{
		self_type retval(val + n);
		return retval;
	}

	/// Addition assignment
	template <typename Distance>
	inline self_type &operator+=(Distance n)
	{
		val += n;
		return *this;
	}

	/// Subtraction
	template <typename Distance>
	inline self_type operator-(Distance n) const
	{
		self_type retval(val - n);
		return retval;
	}

	/// Subtraction assignment
	template <typename Distance>
	inline self_type &operator-=(Distance n)
	{
		val -= n;
		return *this;
	}

	/// Distance
	inline difference_type operator-(self_type other) const
	{
		return val - other.val;
	}

	/// Array subscript
	template <typename Distance>
	inline reference operator[](Distance n) const
	{
		return val + n;
	}

	/// Structure dereference
	inline pointer operator->()
	{
		return &val;
	}

	/// Equal to
	inline bool operator==(const self_type &rhs)
	{
		return (val == rhs.val);
	}

	/// Not equal to
	inline bool operator!=(const self_type &rhs)
	{
		return (val != rhs.val);
	}

	/// ostream operator
	friend std::ostream &operator<<(std::ostream &os, const self_type &itr)
	{
		os << "[" << itr.val << "]";
		return os;
	}
};

//---------------------------------------------------------------------
// MergePath Search
//---------------------------------------------------------------------

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

//---------------------------------------------------------------------
// SpMV verification
//---------------------------------------------------------------------

// Compute reference SpMV y = Ax
template <
	typename ValueT,
	typename OffsetT>
void SpmvGold(
	CsrMatrix<ValueT, OffsetT> &a,
	ValueT *vector_x,
	ValueT *vector_y_in,
	ValueT *vector_y_out,
	ValueT alpha,
	ValueT beta)
{
	for (OffsetT row = 0; row < a.num_rows; ++row)
	{
		ValueT partial = beta * vector_y_in[row];
		for (
			OffsetT offset = a.row_offsets[row];
			offset < a.row_offsets[row + 1];
			++offset)
		{
			partial += alpha * a.values[offset] * vector_x[a.column_indices[offset]];
		}
		vector_y_out[row] = partial;
	}
}

//---------------------------------------------------------------------
// CPU normal omp SpMV
//---------------------------------------------------------------------

template <
	typename ValueT,
	typename OffsetT>
void OmpCsrSpmmT(
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

/**
 * MKL CPU SpMV (specialized for fp64)
 */
template <typename OffsetT>
void MKLCsrmm(
	int num_threads,
	CsrMatrix<double, OffsetT> &a,
	OffsetT *__restrict row_end_offsets, ///< Merge list A (row end-offsets)
	OffsetT *__restrict column_indices,
	double *__restrict values,
	double *__restrict vector_x,
	double *__restrict vector_y_out,
	int num_vectors)
{
	struct matrix_descr A_descr;
	A_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
	sparse_matrix_t csrA;

	mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, a.num_rows, a.num_cols, a.row_offsets, row_end_offsets, a.column_indices, a.values);
	mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, A_descr, SPARSE_LAYOUT_ROW_MAJOR, vector_x, num_vectors, num_vectors, 0.0, vector_y_out, num_vectors);
}

template <
	typename ValueT,
	typename OffsetT>
void OmpMergeCsrmm(
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
void OmpNonzeroSplitCsrmm(
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

//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------

/**
 * Display perf
 */
template <typename ValueT, typename OffsetT>
void DisplayPerf(
	double setup_ms,
	double avg_ms,
	CsrMatrix<ValueT, OffsetT> &csr_matrix,
	int num_vectors)
{
	double nz_throughput, effective_bandwidth;
	size_t total_bytes = (csr_matrix.num_nonzeros * (sizeof(ValueT) * 2 + sizeof(OffsetT))) +
						 (csr_matrix.num_rows * num_vectors) * (sizeof(OffsetT) + sizeof(ValueT));

	nz_throughput = double(csr_matrix.num_nonzeros) * double(num_vectors) / avg_ms / 1.0e6;
	effective_bandwidth = double(total_bytes) / avg_ms / 1.0e6;

	if (!g_quiet)
		printf("fp%d: %.4f setup ms, %.4f avg ms, %.5f gflops, %.3lf effective GB/s\n",
			   int(sizeof(ValueT) * 8),
			   setup_ms,
			   avg_ms,
			   2 * nz_throughput,
			   effective_bandwidth);
	else
		printf("%.5f, %.5f, %.6f, %.3lf, ",
			   setup_ms, avg_ms,
			   2 * nz_throughput,
			   effective_bandwidth);

	fflush(stdout);
}

//---------------------------------------------------------------------
// Block CG parts
//---------------------------------------------------------------------

//---------------------------------------------------------------------
// Helper functions for CG method
//---------------------------------------------------------------------

// Simple SpMV y = Ax
template <
	typename ValueT,
	typename OffsetT>
void SimpleSpmv(
	CsrMatrix<ValueT, OffsetT> &a,
	ValueT *vector_x,
	ValueT *vector_y_out)
{
#pragma omp parallel for
	for (OffsetT row = 0; row < a.num_rows; ++row)
	{
		ValueT partial = 0.0;
		for (OffsetT offset = a.row_offsets[row]; offset < a.row_offsets[row + 1]; ++offset)
		{
			partial += a.values[offset] * vector_x[a.column_indices[offset]];
		}
		vector_y_out[row] = partial;
	}
}

// dot product: returns x^T * y
template <typename ValueT, typename OffsetT>
ValueT dot(OffsetT n, const ValueT *x, const ValueT *y)
{
	ValueT result = 0.0;
#pragma omp parallel for reduction(+ : result)
	for (OffsetT i = 0; i < n; ++i)
	{
		result += x[i] * y[i];
	}
	return result;
}

// axpy: y = y + a*x
template <typename ValueT, typename OffsetT>
void axpy(OffsetT n, ValueT a, const ValueT *x, ValueT *y)
{
#pragma omp parallel for
	for (OffsetT i = 0; i < n; ++i)
	{
		y[i] += a * x[i];
	}
}

//---------------------------------------------------------------------
// Conjugate Gradient Solver (for single RHS)
//---------------------------------------------------------------------
template <
	typename ValueT,
	typename OffsetT>
int CGSolveSingle(
	CsrMatrix<ValueT, OffsetT> &a,
	const ValueT *b, // RHS vector
	ValueT *x,		 // Solution vector
	int max_iters,
	ValueT tolerance)
{
	OffsetT n = a.num_rows;
	ValueT *r, *p, *Ap;

	// Allocate temporary vectors
	r = (ValueT *)mkl_malloc(sizeof(ValueT) * n, 4096);
	p = (ValueT *)mkl_malloc(sizeof(ValueT) * n, 4096);
	Ap = (ValueT *)mkl_malloc(sizeof(ValueT) * n, 4096);

// Initialize
#pragma omp parallel for
	for (OffsetT i = 0; i < n; ++i)
	{
		x[i] = 0.0;
		r[i] = b[i];
		p[i] = b[i];
	}

	ValueT rs_old = dot(n, r, r);
	ValueT rs_new;
	ValueT b_norm = sqrt(dot(n, b, b));
	if (b_norm == 0.0)
		b_norm = 1.0; // Handle zero b vector

	int iter;
	for (iter = 0; iter < max_iters; ++iter)
	{
		// Ap = A * p
		SimpleSpmv(a, p, Ap);

		// alpha = rs_old / (p' * Ap)
		ValueT alpha = rs_old / dot(n, p, Ap);

		// x = x + alpha * p
		axpy(n, alpha, p, x);

		// r = r - alpha * Ap
		axpy(n, -alpha, Ap, r);

		rs_new = dot(n, r, r);

		// Convergence check
		if (sqrt(rs_new) / b_norm < tolerance)
		{
			iter++; // Count this iteration
			break;
		}

		// p = r + (rs_new / rs_old) * p
		ValueT beta = rs_new / rs_old;
#pragma omp parallel for
		for (OffsetT i = 0; i < n; ++i)
		{
			p[i] = r[i] + beta * p[i];
		}

		rs_old = rs_new;
	}

	mkl_free(r);
	mkl_free(p);
	mkl_free(Ap);

	return iter;
}

/**
 * Run and time the sequential single-RHS CG solver
 */
template <
	typename ValueT,
	typename OffsetT>
void TestCGSingleRHS(
	CsrMatrix<ValueT, OffsetT> &a,
	ValueT *b_vectors,
	ValueT *x_solutions,
	int max_iters,
	ValueT tolerance,
	int num_vectors,
	int timing_iterations,
	double &avg_ms,		  // [out] Average time in milliseconds for one run over all vectors
	double &average_iters // [out] Average number of iterations
)
{
	if (!g_quiet)
		printf("\tUsing %d threads on %d procs\n", omp_get_max_threads(), omp_get_num_procs());

	// --- Warmup run ---
	// This run populates caches and gets a representative iteration count.
	for (int i = 0; i < num_vectors; ++i)
	{
		ValueT *b_current = b_vectors + (long long)i * a.num_rows;
		ValueT *x_current = x_solutions + (long long)i * a.num_rows;
		CGSolveSingle(a, b_current, x_current, max_iters, tolerance);
	}

	// --- Timed runs ---
	CpuTimer timer;
	timer.Start();
	int total_iters = 0;
	for (int it = 0; it < timing_iterations; ++it)
	{
		for (int i = 0; i < num_vectors; ++i)
		{
			ValueT *b_current = b_vectors + (long long)i * a.num_rows;
			ValueT *x_current = x_solutions + (long long)i * a.num_rows;
			// Iteration count is not needed here, just the computation for timing.
			int iters = CGSolveSingle(a, b_current, x_current, max_iters, tolerance);
			total_iters += iters;
		}
	}
	timer.Stop();
	double elapsed_ms = timer.ElapsedMillis();

	avg_ms = elapsed_ms / timing_iterations;
	average_iters = double(total_iters) / double(timing_iterations * num_vectors);
}

//---------------------------------------------------------------------
// Helper functions for Multiple-RHS CG method
//---------------------------------------------------------------------

// Enum to specify which SpMM kernel to use
enum SpmmKernel
{
	SIMPLE,
	MERGE,
	NONZERO_SPLIT
};

// dot product for multiple vectors: result[i] = x_i' * y_i
template <typename ValueT, typename OffsetT>
void dot_multiple(OffsetT n, int num_vectors, const ValueT *x, const ValueT *y, ValueT *result)
{
#pragma omp parallel for
	for (int i = 0; i < num_vectors; ++i)
	{
		ValueT local_result = 0.0;
		const ValueT *x_col = x + (long long)i * n;
		const ValueT *y_col = y + (long long)i * n;
		for (OffsetT j = 0; j < n; ++j)
		{
			local_result += x_col[j] * y_col[j];
		}
		result[i] = local_result;
	}
}

// axpy for multiple vectors: y_i = y_i + a[i]*x_i
template <typename ValueT, typename OffsetT>
void axpy_multiple(OffsetT n, int num_vectors, const ValueT *a, const ValueT *x, ValueT *y)
{
#pragma omp parallel for
	for (int i = 0; i < num_vectors; ++i)
	{
		ValueT scalar_a = a[i];
		ValueT *y_col = y + (long long)i * n;
		const ValueT *x_col = x + (long long)i * n;
		for (OffsetT j = 0; j < n; ++j)
		{
			y_col[j] += scalar_a * x_col[j];
		}
	}
}

// p_i = r_i + beta[i] * p_i
template <typename ValueT, typename OffsetT>
void update_p_multiple(OffsetT n, int num_vectors, const ValueT *r, const ValueT *beta, ValueT *p)
{
#pragma omp parallel for
	for (int i = 0; i < num_vectors; ++i)
	{
		ValueT scalar_beta = beta[i];
		const ValueT *r_col = r + (long long)i * n;
		ValueT *p_col = p + (long long)i * n;
		for (OffsetT j = 0; j < n; ++j)
		{
			p_col[j] = r_col[j] + scalar_beta * p_col[j];
		}
	}
}

//---------------------------------------------------------------------
// Conjugate Gradient Solver (for multiple RHS using SpMM)
//---------------------------------------------------------------------
template <
	typename ValueT,
	typename OffsetT>
int CGSolveMultiple(
	CsrMatrix<ValueT, OffsetT> &a,
	const ValueT *B, // RHS vectors (num_rows x num_vectors)
	ValueT *X,		 // Solution vectors (num_rows x num_vectors)
	int num_vectors,
	int max_iters,
	ValueT tolerance,
	SpmmKernel kernel_type)
{
	OffsetT n = a.num_rows;
	ValueT *R, *P, *AP;
	ValueT *vector_x_row_major_dummy = nullptr; // For SpMM interface

	// Allocate temporary matrices
	R = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096);
	P = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096);
	AP = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096);

	ValueT *P_row_major = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096);

	// Allocate temporary scalar arrays
	ValueT *alpha = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	ValueT *beta = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	ValueT *rs_old = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	ValueT *rs_new = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	ValueT *pAp = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	ValueT *b_norms = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	bool *converged = new bool[num_vectors];

// Initialize
#pragma omp parallel for
	for (long long i = 0; i < (long long)n * num_vectors; ++i)
	{
		X[i] = 0.0;
		R[i] = B[i];
		P[i] = B[i];
	}

	int num_converged = 0;
	dot_multiple(n, num_vectors, B, B, b_norms);
#pragma omp parallel for
	for (int i = 0; i < num_vectors; ++i)
	{
		b_norms[i] = sqrt(b_norms[i]);
		if (b_norms[i] == 0.0)
			b_norms[i] = 1.0;
		converged[i] = false;
	}

	dot_multiple(n, num_vectors, R, R, rs_old);

	int iter;
	for (iter = 0; iter < max_iters; ++iter)
	{
		// AP = A * P using the specified SpMM kernel
		memset(AP, 0, sizeof(ValueT) * n * num_vectors);
		switch (kernel_type)
		{
		case SIMPLE:
			OmpCsrSpmmT(g_omp_threads, a, P, AP, num_vectors, P_row_major);
			break;
		case MERGE:
			OmpMergeCsrmm(g_omp_threads, a, a.row_offsets + 1, a.column_indices, a.values, P, AP, num_vectors, P_row_major);
			break;
		case NONZERO_SPLIT:
			OmpNonzeroSplitCsrmm(g_omp_threads, a, a.row_offsets + 1, a.column_indices, a.values, P, AP, num_vectors, P_row_major);
			break;
		}

		dot_multiple(n, num_vectors, P, AP, pAp);

#pragma omp parallel for
		for (int i = 0; i < num_vectors; ++i)
		{
			if (!converged[i])
			{
				alpha[i] = rs_old[i] / pAp[i];
			}
			else
			{
				alpha[i] = 0.0; // No update for converged vectors
			}
		}

		// X = X + alpha * P
		axpy_multiple(n, num_vectors, alpha, P, X);
// R = R - alpha * AP
#pragma omp parallel for
		for (int i = 0; i < num_vectors; ++i)
			alpha[i] = -alpha[i]; // Negate alpha for subtraction
		axpy_multiple(n, num_vectors, alpha, AP, R);

		dot_multiple(n, num_vectors, R, R, rs_new);

		// Convergence check
		num_converged = 0;
#pragma omp parallel for reduction(+ : num_converged)
		for (int i = 0; i < num_vectors; ++i)
		{
			if (!converged[i])
			{
				if (sqrt(rs_new[i]) / b_norms[i] < tolerance)
				{
					converged[i] = true;
				}
			}
			if (converged[i])
				num_converged++;
		}

		if (num_converged == num_vectors)
		{
			iter++;
			break;
		}

// beta = rs_new / rs_old
// p = r + beta * p
#pragma omp parallel for
		for (int i = 0; i < num_vectors; ++i)
		{
			if (!converged[i])
			{
				beta[i] = rs_new[i] / rs_old[i];
			}
			else
			{
				beta[i] = 0.0;
			}
		}
		update_p_multiple(n, num_vectors, R, beta, P);

#pragma omp parallel for
		for (int i = 0; i < num_vectors; ++i)
			rs_old[i] = rs_new[i];
	}

	// Cleanup
	mkl_free(R);
	mkl_free(P);
	mkl_free(AP);
	mkl_free(P_row_major);
	mkl_free(alpha);
	mkl_free(beta);
	mkl_free(rs_old);
	mkl_free(rs_new);
	mkl_free(pAp);
	mkl_free(b_norms);
	delete[] converged;

	return iter;
}

/**
 * Run and time the multiple-RHS CG solver using SpMM
 */
template <
	typename ValueT,
	typename OffsetT>
void TestCGMultipleRHS(
	CsrMatrix<ValueT, OffsetT> &a,
	ValueT *b_vectors,
	ValueT *x_solutions,
	int max_iters,
	ValueT tolerance,
	int num_vectors,
	int timing_iterations,
	SpmmKernel kernel_type,
	double &avg_ms,		  // [out] Average time in milliseconds for one run
	double &average_iters // [out] Average iterations for one run
)
{
	if (!g_quiet)
		printf("\tUsing %d threads on %d procs\n", omp_get_max_threads(), omp_get_num_procs());

	// --- Warmup run ---
	CGSolveMultiple(a, b_vectors, x_solutions, num_vectors, max_iters, tolerance, kernel_type);

	// --- Timed runs ---
	CpuTimer timer;
	timer.Start();
	int total_iters = 0;
	for (int it = 0; it < timing_iterations; ++it)
	{
		int iters = CGSolveMultiple(a, b_vectors, x_solutions, num_vectors, max_iters, tolerance, kernel_type);
		total_iters += iters;
	}
	timer.Stop();
	double elapsed_ms = timer.ElapsedMillis();

	avg_ms = elapsed_ms / timing_iterations;
	average_iters = total_iters / timing_iterations;
}

//---------------------------------------------------------------------
// Preconditioner Helper Functions
//---------------------------------------------------------------------

/**
 * Transposes a CSR matrix.
 */
template <typename ValueT, typename OffsetT>
void TransposeCsr(
	const CsrMatrix<ValueT, OffsetT> &in,
	CsrMatrix<ValueT, OffsetT> &out)
{
	out.num_rows = in.num_cols;
	out.num_cols = in.num_rows;
	out.num_nonzeros = in.num_nonzeros;

	out.row_offsets = (OffsetT *)mkl_malloc(sizeof(OffsetT) * (out.num_rows + 1), 4096);
	out.column_indices = (OffsetT *)mkl_malloc(sizeof(OffsetT) * out.num_nonzeros, 4096);
	out.values = (ValueT *)mkl_malloc(sizeof(ValueT) * out.num_nonzeros, 4096);

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
bool IncompleteCholesky(
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
void ForwardSolveMultiple(
	const CsrMatrix<ValueT, OffsetT> &l,
	const ValueT *b,
	ValueT *x,
	int num_vectors)
{
	OffsetT n = l.num_rows;
#pragma omp parallel for
	for (int vec_idx = 0; vec_idx < num_vectors; ++vec_idx)
	{
		ValueT *x_col = x + (long long)vec_idx * n;
		const ValueT *b_col = b + (long long)vec_idx * n;

		for (OffsetT i = 0; i < n; ++i)
		{
			ValueT sum = 0.0;
			OffsetT diag_offset = 0;
			for (OffsetT j_offset = l.row_offsets[i]; j_offset < l.row_offsets[i + 1]; ++j_offset)
			{
				OffsetT j = l.column_indices[j_offset];
				if (i == j)
				{
					diag_offset = j_offset;
					continue;
				}
				sum += l.values[j_offset] * x_col[j];
			}
			x_col[i] = (b_col[i] - sum) / l.values[diag_offset];
		}
	}
}

/**
 * Multiple-RHS Backward substitution: Solves L^T * X = B
 * L is a lower triangular matrix.
 */
template <typename ValueT, typename OffsetT>
void BackwardSolveMultiple(
	const CsrMatrix<ValueT, OffsetT> &l_t,
	const ValueT *b,
	ValueT *x,
	int num_vectors)
{
	OffsetT n = l_t.num_rows;
#pragma omp parallel for
	for (int vec_idx = 0; vec_idx < num_vectors; ++vec_idx)
	{
		ValueT *x_col = x + (long long)vec_idx * n;
		const ValueT *b_col = b + (long long)vec_idx * n;

		for (OffsetT i = n - 1; i >= 0; --i)
		{
			ValueT sum = 0.0;
			ValueT diag_val = 0.0;

			for (OffsetT j_offset = l_t.row_offsets[i]; j_offset < l_t.row_offsets[i + 1]; ++j_offset)
			{
				OffsetT j = l_t.column_indices[j_offset];
				ValueT val = l_t.values[j_offset];

				if (i == j)
				{
					diag_val = val; // L_T[i][i] (== L[i][i])
				}
				else
				{
					sum += val * x_col[j]; // L_T[i][j] * x[j]
				}
			}

			if (diag_val == 0.0)
			{
				x_col[i] = 0.0;
			}
			else
			{
				x_col[i] = (b_col[i] - sum) / diag_val;
			}
		}
	}
}

//---------------------------------------------------------------------
// Preconditioned Conjugate Gradient Solver (for multiple RHS using SpMM)
//---------------------------------------------------------------------
template <
	typename ValueT,
	typename OffsetT>
int PCGSolveMultiple(
	CsrMatrix<ValueT, OffsetT> &a,
	const CsrMatrix<ValueT, OffsetT> &l,		   // Preconditioner factor L
	const CsrMatrix<ValueT, OffsetT> &l_transpose, // Transpose of L
	const ValueT *B,
	ValueT *X,
	int num_vectors,
	int max_iters,
	ValueT tolerance,
	SpmmKernel kernel_type)
{
	OffsetT n = a.num_rows;
	ValueT *R, *P, *AP, *Z;

	R = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096);
	P = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096);
	AP = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096);
	Z = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096); // z = M^-1 * r
	ValueT *temp_y = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096);

	ValueT *P_row_major = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096);

	ValueT *alpha = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	ValueT *beta = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	ValueT *rho_old = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	ValueT *rho_new = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	ValueT *pAp = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	ValueT *b_norms = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	bool *converged = new bool[num_vectors];

// Initialize
#pragma omp parallel for
	for (long long i = 0; i < (long long)n * num_vectors; ++i)
		X[i] = 0.0;

	// R = B - AX (since X is 0, R=B)
	memcpy(R, B, sizeof(ValueT) * n * num_vectors);

	int num_converged = 0;
	dot_multiple(n, num_vectors, B, B, b_norms);
#pragma omp parallel for
	for (int i = 0; i < num_vectors; ++i)
	{
		b_norms[i] = sqrt(b_norms[i]);
		if (b_norms[i] == 0.0)
			b_norms[i] = 1.0;
		converged[i] = false;
	}

	// Preconditioner solve: Z = M^-1 * R
	ForwardSolveMultiple(l, R, temp_y, num_vectors);
	// Note: The simple backward solve is very slow. A pre-transposed L is recommended.
	BackwardSolveMultiple(l_transpose, temp_y, Z, num_vectors);

	// P = Z
	memcpy(P, Z, sizeof(ValueT) * n * num_vectors);

	dot_multiple(n, num_vectors, R, Z, rho_old);

	int iter;
	for (iter = 0; iter < max_iters; ++iter)
	{
		memset(AP, 0, sizeof(ValueT) * n * num_vectors);
		switch (kernel_type)
		{
		case SIMPLE:
			OmpCsrSpmmT(g_omp_threads, a, P, AP, num_vectors, P_row_major);
			break;
		case MERGE:
			OmpMergeCsrmm(g_omp_threads, a, a.row_offsets + 1, a.column_indices, a.values, P, AP, num_vectors, P_row_major);
			break;
		case NONZERO_SPLIT:
			OmpNonzeroSplitCsrmm(g_omp_threads, a, a.row_offsets + 1, a.column_indices, a.values, P, AP, num_vectors, P_row_major);
			break;
		}

		dot_multiple(n, num_vectors, P, AP, pAp);

#pragma omp parallel for
		for (int i = 0; i < num_vectors; ++i)
		{
			if (!converged[i])
				alpha[i] = rho_old[i] / pAp[i];
			else
				alpha[i] = 0.0;
		}

		axpy_multiple(n, num_vectors, alpha, P, X);

#pragma omp parallel for
		for (int i = 0; i < num_vectors; ++i)
			alpha[i] = -alpha[i];
		axpy_multiple(n, num_vectors, alpha, AP, R);

		ValueT *R_norms = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
		dot_multiple(n, num_vectors, R, R, R_norms);

		num_converged = 0;
#pragma omp parallel for reduction(+ : num_converged)
		for (int i = 0; i < num_vectors; ++i)
		{
			if (!converged[i] && (sqrt(R_norms[i]) / b_norms[i] < tolerance))
			{
				converged[i] = true;
			}
			if (converged[i])
				num_converged++;
		}
		mkl_free(R_norms);

		if (num_converged == num_vectors)
		{
			iter++;
			break;
		}

		ForwardSolveMultiple(l, R, temp_y, num_vectors);
		BackwardSolveMultiple(l_transpose, temp_y, Z, num_vectors);

		dot_multiple(n, num_vectors, R, Z, rho_new);

#pragma omp parallel for
		for (int i = 0; i < num_vectors; ++i)
		{
			if (!converged[i])
				beta[i] = rho_new[i] / rho_old[i];
			else
				beta[i] = 0.0;
		}

		update_p_multiple(n, num_vectors, Z, beta, P);

		memcpy(rho_old, rho_new, sizeof(ValueT) * num_vectors);
	}

	mkl_free(R);
	mkl_free(P);
	mkl_free(AP);
	mkl_free(Z);
	mkl_free(P_row_major);
	mkl_free(temp_y);
	mkl_free(alpha);
	mkl_free(beta);
	mkl_free(rho_old);
	mkl_free(rho_new);
	mkl_free(pAp);
	mkl_free(b_norms);
	delete[] converged;

	return iter;
}

/**
 * Run and time the multiple-RHS PCG solver
 */
template <
	typename ValueT,
	typename OffsetT>
void TestPCGMultipleRHS(
	CsrMatrix<ValueT, OffsetT> &a,
	const CsrMatrix<ValueT, OffsetT> &l,
	const CsrMatrix<ValueT, OffsetT> &l_transpose,
	ValueT *b_vectors,
	ValueT *x_solutions,
	int max_iters,
	ValueT tolerance,
	int num_vectors,
	int timing_iterations,
	SpmmKernel kernel_type,
	double &avg_ms,
	double &average_iters)
{
	PCGSolveMultiple(a, l, l_transpose, b_vectors, x_solutions, num_vectors, max_iters, tolerance, kernel_type);

	CpuTimer timer;
	timer.Start();
	int total_iters = 0;
	for (int it = 0; it < timing_iterations; ++it)
	{
		int iter = PCGSolveMultiple(a, l, l_transpose, b_vectors, x_solutions, num_vectors, max_iters, tolerance, kernel_type);
		total_iters += iter;
	}
	timer.Stop();
	double elapsed_ms = timer.ElapsedMillis();

	avg_ms = elapsed_ms / timing_iterations;
	average_iters = static_cast<double>(total_iters) / timing_iterations;
}

/**
 * Run CG tests
 */
template <
	typename ValueT,
	typename OffsetT>
void RunCgTests(
	const std::string &mtx_filename,
	int max_iters,
	ValueT tolerance,
	int num_vectors,
	CommandLineArgs &args)
{
	// Initialize matrix in COO form
	CooMatrix<ValueT, OffsetT> coo_matrix;

	if (!mtx_filename.empty())
	{
		// Parse matrix market file
		coo_matrix.InitMarket(mtx_filename, 1.0, !g_quiet);

		if ((coo_matrix.num_rows == 1) || (coo_matrix.num_cols == 1) || (coo_matrix.num_nonzeros == 1))
		{
			if (!g_quiet)
				printf("Trivial dataset\n");
			exit(0);
		}
		printf("%s, ", mtx_filename.c_str());
		fflush(stdout);
	}
	else
	{
		fprintf(stderr, "No graph type specified.\n");
		exit(1);
	}

	CsrMatrix<ValueT, OffsetT> csr_matrix(coo_matrix);
	coo_matrix.Clear();

	// Display matrix info
	csr_matrix.Stats().Display(!g_quiet);
	if (!g_quiet)
	{
		printf("\n");
		csr_matrix.DisplayHistogram();
		printf("\n");
		if (g_verbose2)
			csr_matrix.Display();
		printf("\n");
	}
	fflush(stdout);

	int timing_iterations = std::min(1000ull, std::max(10ull, ((16ull << 30) / (csr_matrix.num_nonzeros * num_vectors))));
	timing_iterations = 10;

	// Allocate vectors for multiple RHS
	ValueT *b_vectors = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows * num_vectors, 4096);
	ValueT *x_solutions = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows * num_vectors, 4096);

	// Initialize RHS vectors with random values
	srand(12345);
#pragma omp parallel for
	for (long long i = 0; i < (long long)csr_matrix.num_rows * num_vectors; ++i)
		b_vectors[i] = static_cast<ValueT>(rand()) / static_cast<ValueT>(RAND_MAX);

	// --- Test Execution & Display Performance ---
	double avg_ms;
	double average_iters;
	double gflops;
	double flops_per_iter_single = 2.0 * csr_matrix.num_nonzeros + 10.0 * csr_matrix.num_rows;
	double flops_per_iter_multi = flops_per_iter_single * num_vectors;

	// --- Test 1: Sequential Single-RHS CG --- This process takes too long compared to multi-RHS CG
	// printf("\n\n--- 1. CG (Sequential Single-RHS) ---\n");
	// TestCGSingleRHS(csr_matrix, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, avg_ms, average_iters);
	// gflops = (flops_per_iter_single * average_iters) / (avg_ms / 1000.0) / 1e9;
	// printf("Avg time: %8.3f ms, Avg iters: %6.1f, Overall GFLOPS: %6.2f\n",
	// 	   avg_ms, average_iters, gflops);

	// --- Test 2: Multiple-RHS CG with SpMM ---
	// Simple SpMM
	printf("\n--- 2. CG (Multiple-RHS w/ Simple SpMM) ---\n");
	TestCGMultipleRHS(csr_matrix, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, SIMPLE, avg_ms, average_iters);
	gflops = (flops_per_iter_multi * average_iters) / (avg_ms / 1000.0) / 1e9;
	printf("Avg time: %8.3f ms, Iters: %6.1f, Overall GFLOPS: %6.2f\n", avg_ms, average_iters, gflops);

	// Merge-based SpMM
	printf("\n--- 2. CG (Multiple-RHS w/ Merge-based SpMM) ---\n");
	TestCGMultipleRHS(csr_matrix, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, MERGE, avg_ms, average_iters);
	gflops = (flops_per_iter_multi * average_iters) / (avg_ms / 1000.0) / 1e9;
	printf("Avg time: %8.3f ms, Iters: %6.1f, Overall GFLOPS: %6.2f\n", avg_ms, average_iters, gflops);

	// Nonzero-splitting SpMM
	printf("\n--- 2. CG (Multiple-RHS w/ Nonzero-split SpMM) ---\n");
	TestCGMultipleRHS(csr_matrix, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, NONZERO_SPLIT, avg_ms, average_iters);
	gflops = (flops_per_iter_multi * average_iters) / (avg_ms / 1000.0) / 1e9;
	printf("Avg time: %8.3f ms, Iters: %6.1f, Overall GFLOPS: %6.2f\n", avg_ms, average_iters, gflops);

	// --- Setup for PCG ---
	printf("\n\n--- 3. Preconditioned CG (IC(0) + Multiple-RHS SpMM) ---\n");
	CsrMatrix<ValueT, OffsetT> L_matrix;
	printf("Performing Incomplete Cholesky factorization...\n");
	CpuTimer ic_timer;
	ic_timer.Start();
	bool ic_success = IncompleteCholesky(csr_matrix, L_matrix);
	ic_timer.Stop();

	if (!ic_success)
	{
		printf("IC factorization failed. Skipping PCG tests.\n");
		// ... (cleanup code) ...
		return;
	}
	printf("IC factorization setup time: %.3f ms\n", ic_timer.ElapsedMillis());

	CsrMatrix<ValueT, OffsetT> L_transpose;
	TransposeCsr(L_matrix, L_transpose);

	// --- Test 3: Multiple-RHS PCG with SpMM ---
	// Note: GFLOPS for PCG is higher due to the preconditioner solve
	double nnz_l = L_matrix.num_nonzeros;
	double flops_per_iter_pcg = (2.0 * csr_matrix.num_nonzeros + 4.0 * nnz_l + 12.0 * csr_matrix.num_rows) * num_vectors;

	// Simple SpMM
	TestPCGMultipleRHS(csr_matrix, L_matrix, L_transpose, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, SIMPLE, avg_ms, average_iters);
	gflops = (flops_per_iter_pcg * average_iters) / (avg_ms / 1000.0) / 1e9;
	printf("\nPCG with Simple SpMM:\n");
	printf("Avg time: %8.3f ms, Iters: %6.1f, Overall GFLOPS: %6.2f\n", avg_ms, average_iters, gflops);

	// Merge-based SpMM
	TestPCGMultipleRHS(csr_matrix, L_matrix, L_transpose, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, MERGE, avg_ms, average_iters);
	gflops = (flops_per_iter_pcg * average_iters) / (avg_ms / 1000.0) / 1e9;
	printf("\nPCG with Merge-based SpMM:\n");
	printf("Avg time: %8.3f ms, Iters: %6.1f, Overall GFLOPS: %6.2f\n", avg_ms, average_iters, gflops);

	// Nonzero-splitting SpMM
	TestPCGMultipleRHS(csr_matrix, L_matrix, L_transpose, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, NONZERO_SPLIT, avg_ms, average_iters);
	gflops = (flops_per_iter_pcg * average_iters) / (avg_ms / 1000.0) / 1e9;
	printf("\nPCG with Nonzero-split SpMM:\n");
	printf("Avg time: %8.3f ms, Iters: %6.1f, Overall GFLOPS: %6.2f\n", avg_ms, average_iters, gflops);

	// --- Teardown ---
	L_matrix.Clear();
	mkl_free(b_vectors);
	mkl_free(x_solutions);
}

/**
 * Main
 */
int main(int argc, char **argv)
{
	// Initialize command line
	CommandLineArgs args(argc, argv);

	std::string mtx_filename;
	int max_iters = 100000;
	double tolerance = 1.0e-3;
	int num_vectors = 32;

	g_verbose = args.CheckCmdLineFlag("v");
	g_verbose2 = args.CheckCmdLineFlag("v2");
	g_quiet = args.CheckCmdLineFlag("quiet");
	args.GetCmdLineArgument("mtx", mtx_filename);
	args.GetCmdLineArgument("threads", g_omp_threads);
	args.GetCmdLineArgument("num_vectors", num_vectors);
	args.GetCmdLineArgument("max_iters", max_iters);
	args.GetCmdLineArgument("tolerance", tolerance);

	printf("Args parsed mtx_filename = [%s]\n", mtx_filename.c_str());

	// Check if matrix is specified
	if (mtx_filename.empty())
	{
		fprintf(stderr, "Please specify a matrix file with -mtx=<filename>\n");
		exit(1);
	}

	// Set number of threads
	if (g_omp_threads != -1)
	{
		omp_set_num_threads(g_omp_threads);
	}

	RunCgTests<double, int>(mtx_filename, max_iters, tolerance, num_vectors, args);

	printf("\n");

	return 0;
}
