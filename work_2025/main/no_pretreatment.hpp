#ifndef NO_PRETREATMENT_HPP
#define NO_PRETREATMENT_HPP

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

#include "../../sparse_matrix.h"
#include "../../utils.h"
#include "../hyper_parameters.hpp"
#include "../types.hpp"
#include "../spmm/sample.hpp"
#include "../spmm/row_splitting.hpp"
#include "../spmm/merge_based.hpp"
#include "../spmm/nonzero_splitting.hpp"
#include "../cg/utils_multiple.hpp"

//----------\-----------------------------------------------------------
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

	// Allocate temporary matrices
	R = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096);
	P = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096);
	AP = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096);

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
			OmpCsrSpmmT(g_omp_threads, a, P, AP, num_vectors);
			break;
		case MERGE:
			OmpMergeCsrmm(g_omp_threads, a, a.row_offsets + 1, a.column_indices, a.values, P, AP, num_vectors);
			break;
		case NONZERO_SPLIT:
			OmpNonzeroSplitCsrmm(g_omp_threads, a, a.row_offsets + 1, a.column_indices, a.values, P, AP, num_vectors);
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
		int num_converged = 0;
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
	double &min_ms,			// [out] Minimum time in milliseconds for one run
	double &iters_of_min_ms // [out] Number of iterations for the minimum time run
)
{
	if (!g_quiet)
		printf("\tUsing %d threads on %d procs\n", omp_get_max_threads(), omp_get_num_procs());

	// --- Warmup run ---
	for (int it = 0; it < timing_iterations; ++it)
	{
		if (!g_quiet)
			printf("Warmup iteration %d/%d\n", it + 1, timing_iterations);
		fflush(stdout);
		CGSolveMultiple(a, b_vectors, x_solutions, num_vectors, max_iters, tolerance, kernel_type);
	}

	// --- Timed runs ---
	CpuTimer timer;
	min_ms = std::numeric_limits<double>::max();
	iters_of_min_ms = 0;
	for (int it = 0; it < timing_iterations; ++it)
	{
		if (!g_quiet)
		{
			printf("Timed iteration %d/%d\n", it + 1, timing_iterations);
			fflush(stdout);
		}
		timer.Start();
		int iters = CGSolveMultiple(a, b_vectors, x_solutions, num_vectors, max_iters, tolerance, kernel_type);
		timer.Stop();
		double elapsed_ms = timer.ElapsedMillis();
		if (!g_quiet)
			printf("\tTime: %.3f ms (%d iterations)\n", elapsed_ms, iters);
		if (elapsed_ms < min_ms)
		{
			min_ms = elapsed_ms;
			iters_of_min_ms = iters;
		}
	}
}

#endif // NO_PRETREATMENT_HPP