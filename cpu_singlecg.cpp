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

#include "work_2025/main/single_strategy.hpp"

/**
 * Run CG tests (Single Strategy)
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

	// Timing iterations setting (can be adjusted)
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
	double total_iters; // Changed: This will hold the SUM of iterations for all vectors
	double gflops;

	// FLOPs per iteration for a SINGLE vector (2*NNZ + 10*N approx)
	double flops_per_iter_single = 2.0 * csr_matrix.num_nonzeros + 10.0 * csr_matrix.num_rows;

	// --- Test: Single Strategy (Loop over vectors) ---
	if (!g_quiet)
		printf("\n--- CG (Sequential Single-RHS Strategy) ---\n");

	// 変更: TestCGSolveSingle を呼び出し
	TestCGSolveSingle(csr_matrix, b_vectors, x_solutions, max_iters, tolerance, num_vectors, timing_iterations, min_ms, total_iters);

	// Calculate GFLOPS: (FLOPs/iter/vec * Total_Iters_All_Vecs) / Time
	gflops = (flops_per_iter_single * total_iters) / (min_ms / 1000.0) / 1e9;

	printf("Min time: %8.3f ms, Total Iters: %8.0f, Overall GFLOPS/s: %6.2f\n", min_ms, total_iters, gflops);

	// --- Teardown ---
	// csr_matrix.Clear();
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
	int max_iters = 10000;
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