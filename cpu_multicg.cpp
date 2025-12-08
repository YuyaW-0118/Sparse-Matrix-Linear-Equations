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
#include "work_2025/hyper_parameters.hpp"
#include "work_2025/types.hpp"
#include "work_2025/spmm/sample.hpp"
#include "work_2025/spmm/row_splitting.hpp"
#include "work_2025/spmm/merge_based.hpp"
#include "work_2025/spmm/nonzero_splitting.hpp"
#include "work_2025/cg/utils_multiple.hpp"
#include "work_2025/cg/incomplete_cholesky_decomp.hpp"

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
		CGSolveMultiple(a, b_vectors, x_solutions, num_vectors, max_iters, tolerance, kernel_type);

	// --- Timed runs ---
	CpuTimer timer;
	min_ms = std::numeric_limits<double>::max();
	iters_of_min_ms = 0;
	for (int it = 0; it < timing_iterations; ++it)
	{
		timer.Start();
		int iters = CGSolveMultiple(a, b_vectors, x_solutions, num_vectors, max_iters, tolerance, kernel_type);
		timer.Stop();
		double elapsed_ms = timer.ElapsedMillis();
		if (elapsed_ms < min_ms)
		{
			min_ms = elapsed_ms;
			iters_of_min_ms = iters;
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
	for (int it = 0; it < timing_iterations; ++it)
	{
		// Warmup run
		PCGSolveMultiple(a, l, l_transpose, b_vectors, x_solutions, num_vectors, max_iters, tolerance, kernel_type);
	}

	CpuTimer timer;
	min_ms = std::numeric_limits<double>::max();
	iters_of_min_ms = 0;
	for (int it = 0; it < timing_iterations; ++it)
	{
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

	int timing_iterations = 32;

	// Allocate vectors for multiple RHS
	ValueT *b_vectors = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows * num_vectors, 4096);
	ValueT *x_solutions = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows * num_vectors, 4096);

	// Initialize RHS vectors with random values
	srand(time(NULL));
	for (long long i = 0; i < (long long)csr_matrix.num_rows * num_vectors; ++i)
		b_vectors[i] = static_cast<ValueT>(rand()) / static_cast<ValueT>(RAND_MAX);

	// --- Test Execution & Display Performance ---
	double min_ms;
	double iters_of_min_ms;
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
	TestCGMultipleRHS(csr_matrix, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, SpmmKernel::SIMPLE, min_ms, iters_of_min_ms);
	gflops = (flops_per_iter_multi * iters_of_min_ms) / (min_ms / 1000.0) / 1e9;
	printf("Min time: %8.3f ms, Iters: %6.1f, Overall GFLOPS/s: %6.2f\n", min_ms, iters_of_min_ms, gflops);

	// Nonzero-splitting SpMM
	printf("\n--- 2. CG (Multiple-RHS w/ Nonzero-split SpMM) ---\n");
	TestCGMultipleRHS(csr_matrix, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, SpmmKernel::NONZERO_SPLIT, min_ms, iters_of_min_ms);
	gflops = (flops_per_iter_multi * iters_of_min_ms) / (min_ms / 1000.0) / 1e9;
	printf("Min time: %8.3f ms, Iters: %6.1f, Overall GFLOPS/s: %6.2f\n", min_ms, iters_of_min_ms, gflops);

	// Merge-based SpMM
	printf("\n--- 2. CG (Multiple-RHS w/ Merge-based SpMM) ---\n");
	TestCGMultipleRHS(csr_matrix, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, SpmmKernel::MERGE, min_ms, iters_of_min_ms);
	gflops = (flops_per_iter_multi * iters_of_min_ms) / (min_ms / 1000.0) / 1e9;
	printf("Min time: %8.3f ms, Iters: %6.1f, Overall GFLOPS/s: %6.2f\n", min_ms, iters_of_min_ms, gflops);

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
	TestPCGMultipleRHS(csr_matrix, L_matrix, L_transpose, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, SpmmKernel::SIMPLE, min_ms, iters_of_min_ms);
	gflops = (flops_per_iter_pcg * iters_of_min_ms) / (min_ms / 1000.0) / 1e9;
	printf("\nPCG with Simple SpMM:\n");
	printf("Min time: %8.3f ms, Iters: %6.1f, Overall GFLOPS/s: %6.2f\n", min_ms, iters_of_min_ms, gflops);

	// Nonzero-splitting SpMM
	TestPCGMultipleRHS(csr_matrix, L_matrix, L_transpose, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, SpmmKernel::NONZERO_SPLIT, min_ms, iters_of_min_ms);
	gflops = (flops_per_iter_pcg * iters_of_min_ms) / (min_ms / 1000.0) / 1e9;
	printf("\nPCG with Nonzero-split SpMM:\n");
	printf("Min time: %8.3f ms, Iters: %6.1f, Overall GFLOPS/s: %6.2f\n", min_ms, iters_of_min_ms, gflops);

	// Merge-based SpMM
	TestPCGMultipleRHS(csr_matrix, L_matrix, L_transpose, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, SpmmKernel::MERGE, min_ms, iters_of_min_ms);
	gflops = (flops_per_iter_pcg * iters_of_min_ms) / (min_ms / 1000.0) / 1e9;
	printf("\nPCG with Merge-based SpMM:\n");
	printf("Min time: %8.3f ms, Iters: %6.1f, Overall GFLOPS/s: %6.2f\n", min_ms, iters_of_min_ms, gflops);

	// --- Teardown ---
	csr_matrix.Clear();
	L_matrix.Clear();
	L_transpose.Clear();
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
	double tolerance = 1.0e-5;
	int num_vectors = 2048;

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
		fprintf(stderr, "Please specify a matrix file with --mtx=<filename>\n");
		exit(1);
	}

	// Set number of threads
	if (g_omp_threads != -1)
	{
		omp_set_num_threads(g_omp_threads);
	}

	RunCgTests<double, int>(mtx_filename, max_iters, tolerance, num_vectors, args);
	printf("All tests completed.\n");

	return 0;
}
