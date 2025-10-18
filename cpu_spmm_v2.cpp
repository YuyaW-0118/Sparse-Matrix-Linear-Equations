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

bool g_quiet = false;    // Whether to display stats in CSV format
bool g_verbose = false;  // Whether to display output to console
bool g_verbose2 = false; // Whether to display input to console
int g_omp_threads = -1;  // Number of openMP threads
int g_expected_calls = 1000000;
bool g_input_row_major = true;
bool g_output_row_major = true;

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
    typedef CountingInputIterator self_type;                   ///< My own type
    typedef OffsetT difference_type;                           ///< Type to express the result of subtracting one iterator from another
    typedef ValueType value_type;                              ///< The type of the element the iterator can point to
    typedef ValueType *pointer;                                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType reference;                               ///< The type of a reference to an element the iterator can point to
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
    OffsetT diagonal,             ///< [in]The diagonal to search
    AIteratorT a,                 ///< [in]List A
    BIteratorT b,                 ///< [in]List B
    OffsetT a_len,                ///< [in]Length of A
    OffsetT b_len,                ///< [in]Length of B
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
    int xt_index = 0;

    if (!g_input_row_major)
    {
#pragma omp parallel for schedule(static) num_threads(num_threads) private(xt_index)
        for (int i = 0; i < num_cols; i++)
        {
            for (int j = i; j < num_cols * num_vectors; j += num_cols)
            {
                vector_x_row_major[xt_index] = vector_x[j];
                xt_index += 1;
            }
        }
    }

#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (OffsetT row = 0; row < a.num_rows; ++row)
    {
        ValueT partial[num_vectors] = {0.0};
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
 * Run OmpCsrSpmm
 */

template <
    typename ValueT,
    typename OffsetT>
float TestOmpCsrSpmmT(
    CsrMatrix<ValueT, OffsetT> &a,
    ValueT *vector_x,
    ValueT *reference_vector_y_out,
    ValueT *vector_y_out,
    int timing_iterations,
    float &setup_ms,
    int num_vectors,
    ValueT *vector_x_row_major)
{
    setup_ms = 0.0;

    if (g_omp_threads == -1)
        g_omp_threads = omp_get_num_procs();
    int num_threads = g_omp_threads;

    if (!g_quiet)
        printf("\tUsing %d threads on %d procs\n", g_omp_threads, omp_get_num_procs());

    // Warmup/correctness
    memset(vector_y_out, -1, sizeof(ValueT) * a.num_rows * num_vectors);
    OmpCsrSpmmT(g_omp_threads, a, vector_x, vector_y_out, num_vectors, vector_x_row_major);
    if (!g_quiet)
    {
        // Check answer
        int compare = CompareResults(reference_vector_y_out, vector_y_out, a.num_rows, true);
        printf("\t%s\n", compare ? "FAIL" : "PASS");
        fflush(stdout);
        if (g_input_row_major || g_output_row_major)
        {
            for (int i = 0; i < a.num_rows; i++)
            {
                ValueT diff = reference_vector_y_out[i] - vector_y_out[i * num_vectors];
                if (diff > 0.000001 || diff < -0.0000001)
                {
                    printf("%lf, %lf\n", reference_vector_y_out[i], vector_y_out[i * num_vectors]);
                }
            }
        }
    }

    // Re-populate caches, etc.
    for (int it = 0; it < timing_iterations; ++it)
    {
        OmpCsrSpmmT(g_omp_threads, a, vector_x, vector_y_out, num_vectors, vector_x_row_major);
    }

    // Timing
    float elapsed_ms = 0.0;
    CpuTimer timer;
    timer.Start();
    for (int it = 0; it < timing_iterations; ++it)
    {
        OmpCsrSpmmT(g_omp_threads, a, vector_x, vector_y_out, num_vectors, vector_x_row_major);
    }
    timer.Stop();
    elapsed_ms += timer.ElapsedMillis();

    return elapsed_ms / timing_iterations;
}

//---------------------------------------------------------------------
// MKL SpMV
//---------------------------------------------------------------------
/**
 * MKL CPU SpMV (specialized for fp32)
 */
template <typename OffsetT>
void MKLCsrmm(
    int num_threads,
    CsrMatrix<float, OffsetT> &a,
    OffsetT *__restrict row_end_offsets, ///< Merge list A (row end-offsets)
    OffsetT *__restrict column_indices,
    float *__restrict values,
    float *__restrict vector_x,
    float *__restrict vector_y_out,
    int num_vectors)
{
    struct matrix_descr A_descr;
    A_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    sparse_matrix_t csrA;

    mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, a.num_rows, a.num_cols, a.row_offsets, row_end_offsets, a.column_indices, a.values);
    mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, csrA, A_descr, SPARSE_LAYOUT_ROW_MAJOR, vector_x, num_vectors, num_vectors, 0.0, vector_y_out, num_vectors);
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

/**
 * Run MKL CsrMV
 */
template <
    typename ValueT,
    typename OffsetT>
float TestMKLCsrmm(
    CsrMatrix<ValueT, OffsetT> &a,
    ValueT *vector_x,
    ValueT *reference_vector_y_out,
    ValueT *vector_y_out,
    int timing_iterations,
    float &setup_ms,
    int num_vectors)
{
    setup_ms = 0.0;

    // Warmup/correctness
    memset(vector_y_out, -1, sizeof(ValueT) * a.num_rows * num_vectors);
    MKLCsrmm(g_omp_threads, a, a.row_offsets + 1, a.column_indices, a.values, vector_x, vector_y_out, num_vectors);
    if (!g_quiet)
    {
        // Check answer
        int compare = CompareResults(reference_vector_y_out, vector_y_out, a.num_rows, true);
        printf("\t%s\n", compare ? "FAIL" : "PASS");
        fflush(stdout);
    }

    // Re-populate caches, etc.
    for (int it = 0; it < timing_iterations; ++it)
    {
        MKLCsrmm(g_omp_threads, a, a.row_offsets + 1, a.column_indices, a.values, vector_x, vector_y_out, num_vectors);
    }

    // Timing
    float elapsed_ms = 0.0;
    CpuTimer timer;
    timer.Start();
    for (int it = 0; it < timing_iterations; ++it)
    {
        MKLCsrmm(g_omp_threads, a, a.row_offsets + 1, a.column_indices, a.values, vector_x, vector_y_out, num_vectors);
    }
    timer.Stop();
    elapsed_ms += timer.ElapsedMillis();

    return elapsed_ms / timing_iterations;
}

//---------------------------------------------------------------------
// CPU merge-based SpMV
//---------------------------------------------------------------------

/**
 * OpenMP CPU merge-based SpMV
 */
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
    OffsetT row_carry_out[256];   // The last row-id each worked on by each thread when it finished its path segment
    ValueT *value_carry_out[256]; // The running total within each thread when it finished its path segment

    int num_cols = a.num_cols;
    int num_rows = a.num_rows;
    int xt_index = 0;

#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int tid = 0; tid < num_threads; tid++)
    {
        // Merge list B (NZ indices)
        CountingInputIterator<OffsetT> nonzero_indices(0);

        OffsetT num_merge_items = a.num_rows + a.num_nonzeros;                        // Merge path total length
        OffsetT items_per_thread = (num_merge_items + num_threads - 1) / num_threads; // Merge items per thread

        // Find starting and ending MergePath coordinates (row-idx, nonzero-idx) for each thread
        int2 thread_coord;
        int2 thread_coord_end;
        int start_diagonal = std::min(items_per_thread * tid, num_merge_items);
        int end_diagonal = std::min(start_diagonal + items_per_thread, num_merge_items);

        MergePathSearch(start_diagonal, row_end_offsets, nonzero_indices, a.num_rows, a.num_nonzeros, thread_coord);
        MergePathSearch(end_diagonal, row_end_offsets, nonzero_indices, a.num_rows, a.num_nonzeros, thread_coord_end);

        // Consume whole rows
        ValueT running_total[num_vectors] = {0.0};
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

            ind = thread_coord.x * num_vectors;
            tmp = vector_y_out + ind;
            for (int i = 0; i < num_vectors; i++)
            {
                tmp[i] = running_total[i];
                running_total[i] = 0.0;
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
        value_carry_out[tid] = running_total;
    }

    // Carry-out fix-up (rows spanning multiple threads)
    int ind;
    ValueT *tmp;
    for (int tid = 0; tid < num_threads - 1; ++tid)
    {
        if (row_carry_out[tid] < a.num_rows)
        {
            ind = row_carry_out[tid] * num_vectors;
            tmp = vector_y_out + ind;
            for (int i = 0; i < num_vectors; i++)
            {
                tmp[i] += value_carry_out[tid][i];
            }
        }
    }
}

/**
 * Run OmpMergeCsrmm
 */
template <
    typename ValueT,
    typename OffsetT>
float TestOmpMergeCsrmm(
    CsrMatrix<ValueT, OffsetT> &a,
    ValueT *vector_x,
    ValueT *reference_vector_y_out,
    ValueT *vector_y_out,
    int timing_iterations,
    float &setup_ms,
    int num_vectors,
    ValueT *vector_x_row_major)
{
    setup_ms = 0.0;

    if (g_omp_threads == -1)
        g_omp_threads = omp_get_num_procs();
    int num_threads = g_omp_threads;

    if (!g_quiet)
        printf("\tUsing %d threads on %d procs\n", g_omp_threads, omp_get_num_procs());

    // Warmup/correctness
    memset(vector_y_out, -1, sizeof(ValueT) * a.num_rows * num_vectors);
    OmpMergeCsrmm(g_omp_threads, a, a.row_offsets + 1, a.column_indices, a.values, vector_x, vector_y_out, num_vectors, vector_x_row_major);
    if (!g_quiet)
    {
        // Check answer
        int compare = CompareResults(reference_vector_y_out, vector_y_out, a.num_rows, true);
        printf("\t%s\n", compare ? "FAIL" : "PASS");
        fflush(stdout);
    }

    // Re-populate caches, etc.
    for (int it = 0; it < timing_iterations; ++it)
    {
        OmpMergeCsrmm(g_omp_threads, a, a.row_offsets + 1, a.column_indices, a.values, vector_x, vector_y_out, num_vectors, vector_x_row_major);
    }

    // Timing
    float elapsed_ms = 0.0;
    CpuTimer timer;
    timer.Start();
    for (int it = 0; it < timing_iterations; ++it)
    {
        OmpMergeCsrmm(g_omp_threads, a, a.row_offsets + 1, a.column_indices, a.values, vector_x, vector_y_out, num_vectors, vector_x_row_major);
    }
    timer.Stop();
    elapsed_ms += timer.ElapsedMillis();

    return elapsed_ms / timing_iterations;
}

template <
    typename AIteratorT,
    typename BIteratorT,
    typename OffsetT,
    typename CoordinateT>
inline void RowPathSearch(
    AIteratorT a,                 ///< [in]List A
    BIteratorT b,                 ///< [in]List B
    OffsetT a_len,                ///< [in]Length of A
    CoordinateT &path_coordinate) ///< [out] (x,y) coordinate where diagonal intersects the merge path
{
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
    OffsetT row_carry_out[256];   // The last row-id each worked on by each thread when it finished its path segment
    ValueT *value_carry_out[256]; // The running total within each thread when it finished its path segment

    int num_cols = a.num_cols;
    int num_rows = a.num_rows;
    int xt_index = 0;

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
        ValueT running_total[num_vectors] = {0.0};
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

            ind = thread_coord.x * num_vectors;
            tmp = vector_y_out + ind;
            for (int i = 0; i < num_vectors; i++)
            {
                tmp[i] = running_total[i];
                running_total[i] = 0.0;
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
        value_carry_out[tid] = running_total;
    }

    // Carry-out fix-up (rows spanning multiple threads)
    int ind;
    ValueT *tmp;
    for (int tid = 0; tid < num_threads - 1; ++tid)
    {
        if (row_carry_out[tid] < a.num_rows)
        {
            ind = row_carry_out[tid] * num_vectors;
            tmp = vector_y_out + ind;
            for (int i = 0; i < num_vectors; i++)
            {
                tmp[i] += value_carry_out[tid][i];
            }
        }
    }
}

template <
    typename ValueT,
    typename OffsetT>
float TestOmpNonzeroSplitCsrmm(
    CsrMatrix<ValueT, OffsetT> &a,
    ValueT *vector_x,
    ValueT *reference_vector_y_out,
    ValueT *vector_y_out,
    int timing_iterations,
    float &setup_ms,
    int num_vectors,
    ValueT *vector_x_row_major)
{
    setup_ms = 0.0;

    if (g_omp_threads == -1)
        g_omp_threads = omp_get_num_procs();
    int num_threads = g_omp_threads;

    if (!g_quiet)
        printf("\tUsing %d threads on %d procs\n", g_omp_threads, omp_get_num_procs());

    // Warmup/correctness
    memset(vector_y_out, -1, sizeof(ValueT) * a.num_rows * num_vectors);
    OmpNonzeroSplitCsrmm(g_omp_threads, a, a.row_offsets + 1, a.column_indices, a.values, vector_x, vector_y_out, num_vectors, vector_x_row_major);
    if (!g_quiet)
    {
        // Check answer
        int compare = CompareResults(reference_vector_y_out, vector_y_out, a.num_rows, true);
        printf("\t%s\n", compare ? "FAIL" : "PASS");
        fflush(stdout);
        if (g_input_row_major == g_output_row_major)
        {
            for (int i = 0; i < a.num_rows; i++)
            {
                ValueT diff = reference_vector_y_out[i] - vector_y_out[i * num_vectors];
                if (diff > 0.000001 || diff < -0.0000001)
                {
                    printf("%lf, %lf, %d\n", reference_vector_y_out[i], vector_y_out[i * num_vectors], i);
                }
            }
        }
    }

    // Re-populate caches, etc.
    for (int it = 0; it < timing_iterations; ++it)
    {
        OmpNonzeroSplitCsrmm(g_omp_threads, a, a.row_offsets + 1, a.column_indices, a.values, vector_x, vector_y_out, num_vectors, vector_x_row_major);
    }

    // Timing
    float elapsed_ms = 0.0;
    CpuTimer timer;
    timer.Start();
    for (int it = 0; it < timing_iterations; ++it)
    {
        OmpNonzeroSplitCsrmm(g_omp_threads, a, a.row_offsets + 1, a.column_indices, a.values, vector_x, vector_y_out, num_vectors, vector_x_row_major);
    }
    timer.Stop();
    elapsed_ms += timer.ElapsedMillis();

    return elapsed_ms / timing_iterations;
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

/**
 * Run tests
 */
template <
    typename ValueT,
    typename OffsetT>
void RunTests(
    ValueT alpha,
    ValueT beta,
    const std::string &mtx_filename,
    int grid2d,
    int grid3d,
    int wheel,
    int dense,
    int timing_iterations,
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
    else if (grid2d > 0)
    {
        // Generate 2D lattice
        printf("grid2d_%d, ", grid2d);
        fflush(stdout);
        coo_matrix.InitGrid2d(grid2d, false);
    }
    else if (grid3d > 0)
    {
        // Generate 3D lattice
        printf("grid3d_%d, ", grid3d);
        fflush(stdout);
        coo_matrix.InitGrid3d(grid3d, false);
    }
    else if (wheel > 0)
    {
        // Generate wheel graph
        printf("wheel_%d, ", grid2d);
        fflush(stdout);
        coo_matrix.InitWheel(wheel);
    }
    else if (dense > 0)
    {
        // Generate dense graph
        OffsetT rows = (1 << 24) / dense; // 16M nnz
        printf("dense_%d_x_%d, ", rows, dense);
        fflush(stdout);
        coo_matrix.InitDense(rows, dense);
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

    //  count max-min row nums
    if (!g_quiet)
    {
        int max = 0;
        int min = 100000000;
        for (int i = 0; i < csr_matrix.num_cols; i++)
        {
            int diff = csr_matrix.row_offsets[i + 1] - csr_matrix.row_offsets[i];
            if (diff > max)
            {
                max = diff;
            }
            if (diff < min)
            {
                min = diff;
            }
        }
        printf("min:%d, max:%d\n", min, max);
    }

    // Determine # of timing iterations (aim to run 16 billion nonzeros through, total)
    if (timing_iterations == -1)
    {
        timing_iterations = std::min(1000ull, std::max(10ull, ((16ull << 30) / (csr_matrix.num_nonzeros * num_vectors))));
        timing_iterations = timing_iterations;
        if (timing_iterations < 2)
        {
            timing_iterations = 3;
        }
        if (!g_quiet)
            printf("\t%d timing iterations\n", timing_iterations);
    }

    // Allocate input and output vectors (if available, use NUMA allocation to force storage on the
    // sockets for performance consistency)
    ValueT *vector_x, *vector_y_in, *reference_vector_y_out, *vector_y_out, *vector_x_row_major;
    if (csr_matrix.IsNumaMalloc())
    {
        vector_x = (ValueT *)numa_alloc_onnode(sizeof(ValueT) * csr_matrix.num_cols * num_vectors, 0);
        vector_y_in = (ValueT *)numa_alloc_onnode(sizeof(ValueT) * csr_matrix.num_rows, 0);
        reference_vector_y_out = (ValueT *)numa_alloc_onnode(sizeof(ValueT) * csr_matrix.num_rows, 0);
        vector_y_out = (ValueT *)numa_alloc_onnode(sizeof(ValueT) * csr_matrix.num_rows * num_vectors, 0);
        vector_x_row_major = (ValueT *)numa_alloc_onnode(sizeof(ValueT) * csr_matrix.num_cols * num_vectors, 0);
    }
    else
    {
        vector_x = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_cols * num_vectors, 4096);
        vector_y_in = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows, 4096);
        reference_vector_y_out = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows, 4096);
        vector_y_out = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows * num_vectors, 4096);
        vector_x_row_major = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_cols * num_vectors, 4096);
    }

    for (int col = 0; col < csr_matrix.num_cols * num_vectors; ++col)
    {
        // vector_x[col] = col / csr_matrix.num_cols + 1.0;
        vector_x[col] = 10.0;
        if (g_input_row_major)
        {
            vector_x_row_major[col] = 10.0;
        }
    }

    for (int row = 0; row < csr_matrix.num_rows; ++row)
        vector_y_in[row] = 1;

    // Compute reference answer
    SpmvGold(csr_matrix, vector_x, vector_y_in, reference_vector_y_out, alpha, beta);

    float avg_ms, setup_ms;

    // Simple SpMMT
    if (!g_quiet)
        printf("\n\n");
    printf("Simple CsrMMT, ");
    fflush(stdout);
    avg_ms = TestOmpCsrSpmmT(csr_matrix, vector_x, reference_vector_y_out, vector_y_out, timing_iterations, setup_ms, num_vectors, vector_x_row_major);
    DisplayPerf(setup_ms, avg_ms, csr_matrix, num_vectors);

    // Merge SpMM
    if (!g_quiet)
        printf("\n\n");
    printf("Merge CsrMM, ");
    fflush(stdout);
    avg_ms = TestOmpMergeCsrmm(csr_matrix, vector_x, reference_vector_y_out, vector_y_out, timing_iterations, setup_ms, num_vectors, vector_x_row_major);
    DisplayPerf(setup_ms, avg_ms, csr_matrix, num_vectors);

    // Row-based SpMM
    if (!g_quiet)
        printf("\n\n");
    printf("nonzero splitting CsrMM, ");
    fflush(stdout);
    avg_ms = TestOmpNonzeroSplitCsrmm(csr_matrix, vector_x, reference_vector_y_out, vector_y_out, timing_iterations, setup_ms, num_vectors, vector_x_row_major);
    DisplayPerf(setup_ms, avg_ms, csr_matrix, num_vectors);

    // MKL SpMM
    if (g_input_row_major == g_output_row_major)
    {
        if (!g_quiet)
            printf("\n\n");
        printf("MKL CsrMM, ");
        fflush(stdout);
        avg_ms = TestMKLCsrmm(csr_matrix, vector_x, reference_vector_y_out, vector_y_out, timing_iterations, setup_ms, num_vectors);
        DisplayPerf(setup_ms, avg_ms, csr_matrix, num_vectors);
    }

    // Cleanup
    if (csr_matrix.IsNumaMalloc())
    {
        if (vector_x)
            numa_free(vector_x, sizeof(ValueT) * csr_matrix.num_cols * num_vectors);
        if (vector_y_in)
            numa_free(vector_y_in, sizeof(ValueT) * csr_matrix.num_rows);
        if (reference_vector_y_out)
            numa_free(reference_vector_y_out, sizeof(ValueT) * csr_matrix.num_rows);
        if (vector_y_out)
            numa_free(vector_y_out, sizeof(ValueT) * csr_matrix.num_rows * num_vectors);
    }
    else
    {
        if (vector_x)
            mkl_free(vector_x);
        if (vector_y_in)
            mkl_free(vector_y_in);
        if (reference_vector_y_out)
            mkl_free(reference_vector_y_out);
        if (vector_y_out)
            mkl_free(vector_y_out);
    }
}

/**
 * Main
 */
int main(int argc, char **argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help"))
    {
        printf(
            "%s "
            "[--quiet] "
            "[--v] "
            "[--threads=<OMP threads>] "
            "[--i=<timing iterations>] "
            "[--fp64 (default) | --fp32] "
            "[--alpha=<alpha scalar (default: 1.0)>] "
            "[--beta=<beta scalar (default: 0.0)>] "
            "\n\t"
            "--mtx=<matrix market file> "
            "\n\t"
            "--dense=<cols>"
            "\n\t"
            "--grid2d=<width>"
            "\n\t"
            "--grid3d=<width>"
            "\n\t"
            "--wheel=<spokes>"
            "\n",
            argv[0]);
        exit(0);
    }

    bool fp32;
    std::string mtx_filename;
    int grid2d = -1;
    int grid3d = -1;
    int wheel = -1;
    int dense = -1;
    int timing_iterations = -1;
    float alpha = 1.0;
    float beta = 0.0;
    int num_vectors = 32;

    g_verbose = args.CheckCmdLineFlag("v");
    g_verbose2 = args.CheckCmdLineFlag("v2");
    g_quiet = args.CheckCmdLineFlag("quiet");
    fp32 = args.CheckCmdLineFlag("fp32");
    args.GetCmdLineArgument("i", timing_iterations);
    args.GetCmdLineArgument("mtx", mtx_filename);
    args.GetCmdLineArgument("grid2d", grid2d);
    args.GetCmdLineArgument("grid3d", grid3d);
    args.GetCmdLineArgument("dense", dense);
    args.GetCmdLineArgument("alpha", alpha);
    args.GetCmdLineArgument("beta", beta);
    args.GetCmdLineArgument("threads", g_omp_threads);
    args.GetCmdLineArgument("num_vectors", num_vectors);

    // Run test(s)
    if (fp32)
    {
        RunTests<float, int>(alpha, beta, mtx_filename, grid2d, grid3d, wheel, dense, timing_iterations, num_vectors, args);
    }
    else
    {
        RunTests<double, int>(alpha, beta, mtx_filename, grid2d, grid3d, wheel, dense, timing_iterations, num_vectors, args);
    }

    printf("\n");

    return 0;
}
