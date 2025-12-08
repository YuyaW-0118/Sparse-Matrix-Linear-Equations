#ifndef SINGLE_HPP
#define SINGLE_HPP

#include <omp.h>
#include <mkl.h>

#include "../../sparse_matrix.h"
#include "../../utils.h"
#include "../hyper_parameters.hpp"

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

#endif // SINGLE_HPP
