#ifndef SINGLE_STRATEGY_HPP
#define SINGLE_STRATEGY_HPP

#include <omp.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <immintrin.h>

#include <mkl.h>

#include "../../sparse_matrix.h"
#include "../../utils.h"
#include "../hyper_parameters.hpp"
#include "../types.hpp"

//---------------------------------------------------------------------
// Basic Kernels (Optimized for Single Vector)
//---------------------------------------------------------------------

/**
 * OpenMP Parallel SpMV: y = A * x
 * Inner loop uses SIMD reduction.
 */
template <typename ValueT, typename OffsetT>
void OmpCsrSpmv(
	const CsrMatrix<ValueT, OffsetT> &a,
	const ValueT *x,
	ValueT *y)
{
	const OffsetT num_rows = a.num_rows;
	const OffsetT *row_offsets = a.row_offsets;
	const int *cols = a.column_indices;
	const ValueT *vals = a.values;

#pragma omp parallel for schedule(static)
	for (OffsetT row = 0; row < num_rows; ++row)
	{
		ValueT sum = 0.0;
		OffsetT start = row_offsets[row];
		OffsetT end = row_offsets[row + 1];

		// Gather pattern is hard for auto-vectorization, but we hint it.
		// For large row average degree, this helps.
#pragma omp simd reduction(+ : sum)
		for (OffsetT i = start; i < end; ++i)
		{
			sum += vals[i] * x[cols[i]];
		}
		y[row] = sum;
	}
}

/**
 * Dot Product: result = x^T * y
 */
template <typename ValueT, typename OffsetT>
ValueT DotSingle(OffsetT n, const ValueT *x, const ValueT *y)
{
	ValueT result = 0.0;
#pragma omp parallel for reduction(+ : result)
	for (OffsetT i = 0; i < n; ++i)
	{
		result += x[i] * y[i];
	}
	return result;
}

/**
 * AXPY: y = a*x + y
 */
template <typename ValueT, typename OffsetT>
void AxpySingle(OffsetT n, ValueT a, const ValueT *x, ValueT *y)
{
#pragma omp parallel for simd
	for (OffsetT i = 0; i < n; ++i)
	{
		y[i] += a * x[i];
	}
}

/**
 * Scale and Update P: p = r + beta * p
 * (Calculated as p[i] = r[i] + beta * p[i])
 */
template <typename ValueT, typename OffsetT>
void UpdatePSingle(OffsetT n, const ValueT *r, ValueT beta, ValueT *p)
{
#pragma omp parallel for simd
	for (OffsetT i = 0; i < n; ++i)
	{
		p[i] = r[i] + beta * p[i];
	}
}

//---------------------------------------------------------------------
// Conjugate Gradient Solver (Single RHS)
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

	// Use MKL malloc for aligned memory allocation (better for SIMD)
	ValueT *r = (ValueT *)mkl_malloc(sizeof(ValueT) * n, 64);
	ValueT *p = (ValueT *)mkl_malloc(sizeof(ValueT) * n, 64);
	ValueT *Ap = (ValueT *)mkl_malloc(sizeof(ValueT) * n, 64);

	// Initialize: x = 0, r = b, p = b
#pragma omp parallel for simd
	for (OffsetT i = 0; i < n; ++i)
	{
		x[i] = 0.0;
		r[i] = b[i];
		p[i] = b[i];
	}

	ValueT rs_old = DotSingle(n, r, r);
	ValueT b_norm = std::sqrt(DotSingle(n, b, b));
	if (b_norm == 0.0)
		b_norm = 1.0;

	int iter = 0;
	for (; iter < max_iters; ++iter)
	{
		// Ap = A * p
		OmpCsrSpmv(a, p, Ap);

		// alpha = rs_old / (p^T * Ap)
		ValueT pAp = DotSingle(n, p, Ap);
		ValueT alpha = rs_old / pAp;

		// x = x + alpha * p
		AxpySingle(n, alpha, p, x);

		// r = r - alpha * Ap
		AxpySingle(n, -alpha, Ap, r);

		ValueT rs_new = DotSingle(n, r, r);

		// Convergence check
		if (std::sqrt(rs_new) / b_norm < tolerance)
		{
			iter++;
			break;
		}

		// p = r + (rs_new / rs_old) * p
		ValueT beta = rs_new / rs_old;
		UpdatePSingle(n, r, beta, p);

		rs_old = rs_new;
	}

	mkl_free(r);
	mkl_free(p);
	mkl_free(Ap);

	return iter;
}

/**
 * Run and time the Single RHS CG solver for multiple vectors (Iteratively)
 * Matches the signature and behavior of TestCGMultipleRHS for comparison.
 */
template <
	typename ValueT,
	typename OffsetT>
void TestCGSolveSingle(
	CsrMatrix<ValueT, OffsetT> &a,
	ValueT *b_vectors,	 // (n x num_vectors) flattened
	ValueT *x_solutions, // (n x num_vectors) flattened
	int max_iters,
	ValueT tolerance,
	int num_vectors,
	int timing_iterations,
	double &min_ms,			// [out] Minimum total time for solving ALL vectors
	double &iters_of_min_ms // [out] Total iterations across all vectors (or avg, depending on usage)
)
{
	if (!g_quiet)
		printf("\t[Single Strategy] Using %d threads on %d procs\n", omp_get_max_threads(), omp_get_num_procs());

	OffsetT n = a.num_rows;

	// --- Warmup run ---
	for (int it = 0; it < timing_iterations; ++it)
	{
		break;
		if (!g_quiet && it == 0)
			printf("\tWarmup...\n");
		for (int v = 0; v < num_vectors; ++v)
		{
			ValueT *b_curr = &b_vectors[v * n];
			ValueT *x_curr = &x_solutions[v * n];
			CGSolveSingle(a, b_curr, x_curr, max_iters, tolerance);
		}
	}

	// --- Timed runs ---
	CpuTimer timer;
	min_ms = std::numeric_limits<double>::max();
	iters_of_min_ms = 0;

	for (int it = 0; it < timing_iterations; ++it)
	{
		timer.Start();
		long long total_iters = 0;

		// Solve for each vector sequentially
		for (int v = 0; v < num_vectors; ++v)
		{
			ValueT *b_curr = &b_vectors[v * n];
			ValueT *x_curr = &x_solutions[v * n];
			total_iters += CGSolveSingle(a, b_curr, x_curr, max_iters, tolerance);
		}

		timer.Stop();
		double elapsed_ms = timer.ElapsedMillis();

		if (!g_quiet)
			printf("\tTime: %.3f ms (Total Iters: %lld)\n", elapsed_ms, total_iters);

		if (elapsed_ms < min_ms)
		{
			min_ms = elapsed_ms;
			iters_of_min_ms = (double)total_iters; // Store total iterations
		}
	}
}

#endif // SINGLE_STRATEGY_HPP