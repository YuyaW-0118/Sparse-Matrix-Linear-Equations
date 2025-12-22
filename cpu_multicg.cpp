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
#include "work_2025/main/no_pretreatment.hpp"
#include "work_2025/main/incomplete_cholesky.hpp"
#include "work_2025/main/sparse_approximate_inverse.hpp"

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

	int timing_iterations = std::clamp((16ull << 30) / (csr_matrix.num_nonzeros * num_vectors), 10ull, 1000ull);
	timing_iterations = 10;
	if (!g_quiet)
		printf("Timing iterations: %d\n for %d non-zeros and %d vectors\n", timing_iterations, csr_matrix.num_nonzeros, num_vectors);

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
	// if (!g_quiet)
	// printf("\n\n--- 1. CG (Sequential Single-RHS) ---\n");
	// TestCGSingleRHS(csr_matrix, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, avg_ms, average_iters);
	// gflops = (flops_per_iter_single * average_iters) / (avg_ms / 1000.0) / 1e9;
	// printf("Avg time: %8.3f ms, Avg iters: %6.1f, Overall GFLOPS: %6.2f\n",
	// 	   avg_ms, average_iters, gflops);

	// // --- Test 2: Multiple-RHS CG with SpMM ---
	// // Simple SpMM
	// if (!g_quiet)
	// 	printf("\n--- 2. CG (Multiple-RHS w/ Simple SpMM) ---\n");
	// TestCGMultipleRHS(csr_matrix, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, SpmmKernel::SIMPLE, min_ms, iters_of_min_ms);
	// gflops = (flops_per_iter_multi * iters_of_min_ms) / (min_ms / 1000.0) / 1e9;
	// 	printf("Min time: %8.3f ms, Iters: %6.1f, Overall GFLOPS/s: %6.2f\n", min_ms, iters_of_min_ms, gflops);

	// Nonzero-splitting SpMM
	if (!g_quiet)
		printf("\n--- 2. CG (Multiple-RHS w/ Nonzero-split SpMM) ---\n");
	TestCGMultipleRHS(csr_matrix, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, SpmmKernel::NONZERO_SPLIT, min_ms, iters_of_min_ms);
	gflops = (flops_per_iter_multi * iters_of_min_ms) / (min_ms / 1000.0) / 1e9;
	printf("Min time: %8.3f ms, Iters: %6.1f, Overall GFLOPS/s: %6.2f\n", min_ms, iters_of_min_ms, gflops);

	// // Merge-based SpMM
	// if (!g_quiet)
	// 	printf("\n--- 2. CG (Multiple-RHS w/ Merge-based SpMM) ---\n");
	// TestCGMultipleRHS(csr_matrix, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, SpmmKernel::MERGE, min_ms, iters_of_min_ms);
	// gflops = (flops_per_iter_multi * iters_of_min_ms) / (min_ms / 1000.0) / 1e9;
	// 	printf("Min time: %8.3f ms, Iters: %6.1f, Overall GFLOPS/s: %6.2f\n", min_ms, iters_of_min_ms, gflops);

	// --- Setup for PCG ---
	if (!g_quiet)
	{
		printf("\n\n--- 3. Preconditioned CG (IC(0) + Multiple-RHS SpMM) ---\n");
		printf("Performing Incomplete Cholesky factorization...\n");
	}
	CsrMatrix<ValueT, OffsetT> L_matrix;
	CpuTimer ic_timer;
	ic_timer.Start();
	bool ic_success = IncompleteCholesky(csr_matrix, L_matrix);
	ic_timer.Stop();

	if (!ic_success)
	{
		printf("IC factorization failed. Skipping PCG tests.\n");
		return;
	}
	if (!g_quiet)
		printf("IC factorization setup time: %.3f ms\n", ic_timer.ElapsedMillis());

	CsrMatrix<ValueT, OffsetT> L_transpose;
	TransposeCsr(L_matrix, L_transpose);

	// --- Test 3: Multiple-RHS PCG with SpMM ---
	// Note: GFLOPS for PCG is higher due to the preconditioner solve
	double nnz_l = L_matrix.num_nonzeros;
	double flops_per_iter_pcg = (2.0 * csr_matrix.num_nonzeros + 4.0 * nnz_l + 12.0 * csr_matrix.num_rows) * num_vectors;

	// // Simple SpMM
	// TestPCGMultipleRHS(csr_matrix, L_matrix, L_transpose, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, SpmmKernel::SIMPLE, min_ms, iters_of_min_ms);
	// gflops = (flops_per_iter_pcg * iters_of_min_ms) / (min_ms / 1000.0) / 1e9;
	// if (!g_quiet)
	// 	printf("\nPCG with Simple SpMM:\n");
	// 	printf("Min time: %8.3f ms, Iters: %6.1f, Overall GFLOPS/s: %6.2f\n", min_ms, iters_of_min_ms, gflops);

	// Nonzero-splitting SpMM
	TestPCGMultipleRHS(csr_matrix, L_matrix, L_transpose, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, SpmmKernel::NONZERO_SPLIT, min_ms, iters_of_min_ms);
	gflops = (flops_per_iter_pcg * iters_of_min_ms) / (min_ms / 1000.0) / 1e9;
	if (!g_quiet)
		printf("\nPCG with Nonzero-split SpMM:\n");
	printf("Min time: %8.3f ms, Iters: %6.1f, Overall GFLOPS/s: %6.2f\n", min_ms, iters_of_min_ms, gflops);

	// // Merge-based SpMM
	// TestPCGMultipleRHS(csr_matrix, L_matrix, L_transpose, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, SpmmKernel::MERGE, min_ms, iters_of_min_ms);
	// gflops = (flops_per_iter_pcg * iters_of_min_ms) / (min_ms / 1000.0) / 1e9;
	// if (!g_quiet)
	// 	printf("\nPCG with Merge-based SpMM:\n");
	// printf("Min time: %8.3f ms, Iters: %6.1f, Overall GFLOPS/s: %6.2f\n", min_ms, iters_of_min_ms, gflops);

	// --- Setup for SPAI ---
	if (!g_quiet)
	{
		printf("\n\n--- 4. Sparse Approximate Inversion (SPAI) ---\n");
		printf("Performing SPAI factorization...\n");
	}
	CsrMatrix<ValueT, OffsetT> spa_matrix;
	CpuTimer spa_timer;
	spa_timer.Start();
	bool spa_success = SparseApproximateInversion(csr_matrix, spa_matrix);
	spa_timer.Stop();
	if (!spa_success)
	{
		printf("SPAI factorization failed. Skipping SPAI tests.\n");
		return;
	}
	printf("SPAI factorization setup time: %.3f ms\n", spa_timer.ElapsedMillis());

	double flops_per_iter_spai = (4.0 * csr_matrix.num_nonzeros + 12.0 * csr_matrix.num_rows) * num_vectors;

	// Simple SpMM
	if (!g_quiet)
		printf("\n--- 4. CG (Multiple-RHS w/ SPAI) ---\n");
	TestCGMultipleSPAI(csr_matrix, spa_matrix, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, SpmmKernel::SIMPLE, min_ms, iters_of_min_ms);
	gflops = (flops_per_iter_spai * iters_of_min_ms) / (min_ms / 1000.0) / 1e9;
	printf("Min time: %8.3f ms, Iters: %6.1f, Overall GFLOPS/s: %6.2f\n", min_ms, iters_of_min_ms, gflops);

	// --- Teardown ---
	csr_matrix.Clear();
	spa_matrix.Clear();
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
	int num_vectors = 32;

	g_verbose = args.CheckCmdLineFlag("v");
	g_verbose2 = args.CheckCmdLineFlag("v2");
	g_quiet = args.CheckCmdLineFlag("quiet");
	args.GetCmdLineArgument("mtx", mtx_filename);
	args.GetCmdLineArgument("threads", g_omp_threads);
	args.GetCmdLineArgument("num_vectors", num_vectors);
	args.GetCmdLineArgument("max_iters", max_iters);
	args.GetCmdLineArgument("tolerance", tolerance);

	if (!g_quiet)
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
	if (!g_quiet)
		printf("All tests completed.\n");

	return 0;
}
