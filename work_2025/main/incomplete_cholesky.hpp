#ifndef INCOMPLETE_CHOLESKY_HPP
#define INCOMPLETE_CHOLESKY_HPP

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
#include "../cg/incomplete_cholesky_decomp.hpp"

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
	double &min_ms,			// [out] Minimum time in milliseconds for one run
	double &iters_of_min_ms // [out] Number of iterations for the minimum time run
)
{
	// --- Warmup run ---
	for (int it = 0; it < timing_iterations; ++it)
	{
		if (!g_quiet)
			printf("Warmup run %d/%d\n", it + 1, timing_iterations);
		fflush(stdout);
		PCGSolveMultiple(a, l, l_transpose, b_vectors, x_solutions, num_vectors, max_iters, tolerance, kernel_type);
	}

	CpuTimer timer;
	min_ms = std::numeric_limits<double>::max();
	iters_of_min_ms = 0;
	for (int it = 0; it < timing_iterations; ++it)
	{
		if (!g_quiet)
		{
			printf("Timed run %d/%d\n", it + 1, timing_iterations);
			fflush(stdout);
		}
		timer.Start();
		int iter = PCGSolveMultiple(a, l, l_transpose, b_vectors, x_solutions, num_vectors, max_iters, tolerance, kernel_type);
		timer.Stop();
		double elapsed_ms = timer.ElapsedMillis();
		if (elapsed_ms < min_ms)
		{
			min_ms = elapsed_ms;
			iters_of_min_ms = iter;
		}
	}
}

#endif // INCOMPLETE_CHOLESKY_HPP