#ifndef SPARSE_APPROXIMATE_INVERSE_HPP
#define SPARSE_APPROXIMATE_INVERSE_HPP

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
#include "../cg/sparse_approximate_inversion.hpp"

template <typename ValueT, typename OffsetT>
int SPAISolveMultiple(
	CsrMatrix<ValueT, OffsetT> &a, // 元の行列 A
	CsrMatrix<ValueT, OffsetT> &m, // 前処理行列 M (SPAI)
	const ValueT *B,			   // 右辺項
	ValueT *X,					   // 解ベクトル
	int num_vectors,
	int max_iters,
	ValueT tolerance,
	SpmmKernel kernel_type)
{
	OffsetT n = a.num_rows;
	ValueT *R, *P, *AP, *Z;

	// Allocate aligned memory for better vectorization
	R = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096);
	P = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096);
	AP = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096);
	Z = (ValueT *)mkl_malloc(sizeof(ValueT) * n * num_vectors, 4096);

	ValueT *alpha = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	ValueT *beta = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	ValueT *rs_old = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	ValueT *rs_new = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	ValueT *pAp = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	ValueT *b_norms = (ValueT *)mkl_malloc(sizeof(ValueT) * num_vectors, 4096);
	bool *converged = new bool[num_vectors];

	// Initialize: X = 0, R = B, P = 0, Z = 0
#pragma omp parallel for
	for (long long i = 0; i < (long long)n * num_vectors; ++i)
	{
		X[i] = 0.0;
		R[i] = B[i];
		P[i] = 0.0;
		Z[i] = 0.0;
	}

	// Calculate norms of B for convergence check
	dot_multiple(n, num_vectors, B, B, b_norms);
#pragma omp parallel for
	for (int i = 0; i < num_vectors; ++i)
	{
		b_norms[i] = sqrt(b_norms[i]);
		if (b_norms[i] == 0.0)
			b_norms[i] = 1.0;
		converged[i] = false;
	}

	// --- Initial Preconditioning: Z = M * R ---
	switch (kernel_type)
	{
	case SIMPLE:
		OmpCsrSpmmT(g_omp_threads, m, R, Z, num_vectors);
		break;
	case MERGE:
		OmpMergeCsrmm(g_omp_threads, m, m.row_offsets + 1, m.column_indices, m.values, R, Z, num_vectors);
		break;
	case NONZERO_SPLIT:
		OmpNonzeroSplitCsrmm(g_omp_threads, m, m.row_offsets + 1, m.column_indices, m.values, R, Z, num_vectors);
		break;
	}

	// P = Z (Initial direction)
#pragma omp parallel for
	for (long long i = 0; i < (long long)n * num_vectors; ++i)
		P[i] = Z[i];

	// rs_old = R * Z
	dot_multiple(n, num_vectors, R, Z, rs_old);

	int iter;
	for (iter = 0; iter < max_iters; ++iter)
	{
		// AP = A * P
		// Note: Even for converged vectors, we compute AP to maintain SIMD efficiency in the kernel.
		// Unnecessary updates will be masked by alpha = 0.
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

		// pAp = P * AP
		dot_multiple(n, num_vectors, P, AP, pAp);

		// Calculate alpha
#pragma omp parallel for
		for (int i = 0; i < num_vectors; ++i)
		{
			if (!converged[i] && pAp[i] != 0.0)
				alpha[i] = rs_old[i] / pAp[i];
			else
				alpha[i] = 0.0;
		}

		// X = X + alpha * P
		axpy_multiple(n, num_vectors, alpha, P, X);

		// R = R - alpha * AP
		// Invert alpha for subtraction using axpy
#pragma omp parallel for
		for (int i = 0; i < num_vectors; ++i)
			alpha[i] = -alpha[i];

		axpy_multiple(n, num_vectors, alpha, AP, R);

		// Check convergence (Calculate R norm)
		// Re-using pAp buffer temporarily to store R*R results to avoid allocating new memory
		dot_multiple(n, num_vectors, R, R, pAp);

		int num_converged = 0;
		double min_not_converged = std::numeric_limits<double>::max();
#pragma omp parallel for reduction(+ : num_converged)
		for (int i = 0; i < num_vectors; ++i)
		{
			if (!converged[i])
			{
				min_not_converged = std::min(min_not_converged, sqrt(pAp[i]) / b_norms[i]);
				if (sqrt(pAp[i]) / b_norms[i] < tolerance)
					converged[i] = true;
			}
			if (converged[i])
				num_converged++;
		}

		if (num_converged == num_vectors)
		{
			iter++;
			break;
		}

		// --- Preconditioning: Z = M * R ---
		switch (kernel_type)
		{
		case SIMPLE:
			OmpCsrSpmmT(g_omp_threads, m, R, Z, num_vectors);
			break;
		case MERGE:
			OmpMergeCsrmm(g_omp_threads, m, m.row_offsets + 1, m.column_indices, m.values, R, Z, num_vectors);
			break;
		case NONZERO_SPLIT:
			OmpNonzeroSplitCsrmm(g_omp_threads, m, m.row_offsets + 1, m.column_indices, m.values, R, Z, num_vectors);
			break;
		}

		// rs_new = R * Z
		dot_multiple(n, num_vectors, R, Z, rs_new);

		// Calculate beta
#pragma omp parallel for
		for (int i = 0; i < num_vectors; ++i)
		{
			if (!converged[i] && rs_old[i] != 0.0)
				beta[i] = rs_new[i] / rs_old[i];
			else
				beta[i] = 0.0;

			rs_old[i] = rs_new[i];
		}

		// P = Z + beta * P
		// Using helper: p = r + beta * p. Here we pass Z as 'r'.
		update_p_multiple(n, num_vectors, Z, beta, P);
	}

	mkl_free(R);
	mkl_free(P);
	mkl_free(AP);
	mkl_free(Z);
	mkl_free(alpha);
	mkl_free(beta);
	mkl_free(rs_old);
	mkl_free(rs_new);
	mkl_free(pAp);
	mkl_free(b_norms);
	delete[] converged;

	return iter;
}

template <
	typename ValueT,
	typename OffsetT>
void TestCGMultipleSPAI(
	CsrMatrix<ValueT, OffsetT> &a,
	CsrMatrix<ValueT, OffsetT> &m,
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
		// SPAISolveMultiple(a, m, b_vectors, x_solutions, num_vectors, max_iters, tolerance, kernel_type);
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
		int iters = SPAISolveMultiple(a, m, b_vectors, x_solutions, num_vectors, max_iters, tolerance, kernel_type);
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

#endif // SPARSE_APPROXIMATE_INVERSE_HPP