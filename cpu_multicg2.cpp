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

/**
 * cpu_multicg2.cpp
 *
 * Benchmark CG solver with 3 SpMM methods (SIMPLE, MERGE, NONZERO_SPLIT)
 * across multiple num_vectors values (2, 4, 8, 16, 32, 64, 128).
 * Outputs GFLOPS results to CSV in data/gflops/.
 */

#include <omp.h>

#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
#include <iomanip>
#include <immintrin.h>
#include <cmath>

#include <mkl.h>

#include "sparse_matrix.h"
#include "utils.h"
#include "work_2025/hyper_parameters.hpp"
#include "work_2025/main/no_pretreatment.hpp"
#include "work_2025/main/incomplete_cholesky.hpp"
#include "work_2025/main/sparse_approximate_inverse.hpp"

template <typename ValueT, typename OffsetT>
ValueT calculate_threshold(const ValueT *b, OffsetT num_rows, ValueT tolerance)
{
	ValueT norm_sq = 0.0;

#pragma omp parallel for reduction(+ : norm_sq)
	for (OffsetT i = 0; i < num_rows; ++i)
	{
		norm_sq += b[i] * b[i];
	}

	return std::sqrt(norm_sq) * tolerance;
}

/**
 * Extract base name from matrix file path
 */
std::string GetMatrixBaseName(const std::string &mtx_filename)
{
	size_t last_slash = mtx_filename.find_last_of("/\\");
	std::string basename = (last_slash == std::string::npos) ? mtx_filename : mtx_filename.substr(last_slash + 1);
	size_t dot_pos = basename.find_last_of('.');
	if (dot_pos != std::string::npos)
	{
		basename = basename.substr(0, dot_pos);
	}
	return basename;
}

/**
 * Get SpMM kernel name string
 */
const char *GetKernelName(SpmmKernel kernel)
{
	switch (kernel)
	{
	case SIMPLE:
		return "SIMPLE";
	case MERGE:
		return "MERGE";
	case NONZERO_SPLIT:
		return "NONZERO_SPLIT";
	default:
		return "UNKNOWN";
	}
}

/**
 * Structure to hold benchmark results
 */
struct BenchmarkResult
{
	std::string matrix_name;
	std::string kernel_name;
	int num_vectors;
	double min_ms;
	double gflops;
	int iterations;
};

/**
 * Run CG benchmark for a specific configuration
 */
template <typename ValueT, typename OffsetT>
BenchmarkResult RunCGBenchmark(
	CsrMatrix<ValueT, OffsetT> &csr_matrix,
	const std::string &matrix_name,
	int num_vectors,
	int max_iters,
	ValueT tolerance,
	int timing_iterations,
	ValueT *b_vectors,
	SpmmKernel kernel_type)
{
	BenchmarkResult result;
	result.matrix_name = matrix_name;
	result.kernel_name = GetKernelName(kernel_type);
	result.num_vectors = num_vectors;

	// Allocate vectors
	ValueT *x_solutions = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows * num_vectors, 4096);

	// Calculate FLOPS per iteration
	double flops_per_iter_single = 2.0 * csr_matrix.num_nonzeros + 10.0 * csr_matrix.num_rows;
	double flops_per_iter_multi = flops_per_iter_single * num_vectors;

	// Run benchmark
	double min_ms;
	double iters_of_min_ms;
	TestCGMultipleRHS(csr_matrix, b_vectors, x_solutions, max_iters, tolerance, num_vectors,
					  timing_iterations, kernel_type, min_ms, iters_of_min_ms, nullptr);

	result.min_ms = min_ms;
	result.iterations = static_cast<int>(iters_of_min_ms);
	result.gflops = (flops_per_iter_multi * iters_of_min_ms) / (min_ms / 1000.0) / 1e9;

	// Cleanup
	mkl_free(b_vectors);
	mkl_free(x_solutions);

	return result;
}

/**
 * Run all benchmarks for a matrix
 */
template <typename ValueT, typename OffsetT>
void RunAllBenchmarks(
	const std::string &mtx_filename,
	int max_iters,
	ValueT tolerance,
	int timing_iterations,
	std::vector<BenchmarkResult> &results)
{
	// Initialize matrix in COO form
	CooMatrix<ValueT, OffsetT> coo_matrix;

	if (!mtx_filename.empty())
	{
		coo_matrix.InitMarket(mtx_filename, 1.0, !g_quiet);

		if ((coo_matrix.num_rows == 1) || (coo_matrix.num_cols == 1) || (coo_matrix.num_nonzeros == 1))
		{
			if (!g_quiet)
				printf("Trivial dataset\n");
			return;
		}
	}
	else
	{
		fprintf(stderr, "No matrix file specified.\n");
		return;
	}

	CsrMatrix<ValueT, OffsetT> csr_matrix(coo_matrix);
	coo_matrix.Clear();

	std::string matrix_name = GetMatrixBaseName(mtx_filename);

	// Display matrix info
	printf("Matrix: %s\n", matrix_name.c_str());
	printf("  Rows: %d, Cols: %d, NNZ: %d\n",
		   csr_matrix.num_rows, csr_matrix.num_cols, csr_matrix.num_nonzeros);

	// Test configurations
	std::vector<int> num_vectors_list = {2, 4, 8, 16, 32, 64, 128};
	std::vector<SpmmKernel> kernels = {SIMPLE, MERGE, NONZERO_SPLIT};

	// Run benchmarks
	for (int num_vectors : num_vectors_list)
	{
		ValueT *b_vectors = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows * num_vectors, 4096);
		// Initialize RHS vectors with random values
		srand(42); // Fixed seed for reproducibility
		for (long long i = 0; i < (long long)csr_matrix.num_rows * num_vectors; ++i)
			b_vectors[i] = static_cast<ValueT>(rand()) / static_cast<ValueT>(RAND_MAX);

		ValueT threshold = calculate_threshold(b_vectors, csr_matrix.num_rows, tolerance);
		fprintf(stderr, "  num_vectors=%d, threshold=%.6Le\n", num_vectors, threshold);

		for (SpmmKernel kernel : kernels)
		{
			if (!g_quiet)
			{
				printf("  Testing: num_vectors=%d, kernel=%s\n",
					   num_vectors, GetKernelName(kernel));
			}

			BenchmarkResult result = RunCGBenchmark(
				csr_matrix, matrix_name, num_vectors, max_iters, threshold,
				timing_iterations, b_vectors, kernel);

			results.push_back(result);

			printf("    %s, L=%d, kernel=%s: %.3f ms, %d iters, %.2f GFLOPS\n",
				   matrix_name.c_str(), num_vectors, GetKernelName(kernel),
				   result.min_ms, result.iterations, result.gflops);
		}
	}
}

/**
 * Save results to CSV
 */
void SaveResultsToCSV(const std::string &filename, const std::vector<BenchmarkResult> &results)
{
	std::ofstream ofs(filename);
	if (!ofs.is_open())
	{
		fprintf(stderr, "Error: Cannot open file %s for writing\n", filename.c_str());
		return;
	}

	// Header
	ofs << "matrix_name,kernel,num_vectors,min_ms,gflops,iterations" << std::endl;

	// Data
	for (const auto &r : results)
	{
		ofs << r.matrix_name << ","
			<< r.kernel_name << ","
			<< r.num_vectors << ","
			<< std::fixed << std::setprecision(3) << r.min_ms << ","
			<< std::fixed << std::setprecision(2) << r.gflops << ","
			<< r.iterations << std::endl;
	}

	ofs.close();
	printf("Results saved to: %s\n", filename.c_str());
}

/**
 * Main
 */
int main(int argc, char **argv)
{
	// Initialize command line
	CommandLineArgs args(argc, argv);

	std::string mtx_filename;
	std::string output_csv;
	int max_iters = 10000;
	double tolerance = 1.0e-5;
	int timing_iterations = 3;

	g_verbose = args.CheckCmdLineFlag("v");
	g_verbose2 = args.CheckCmdLineFlag("v2");
	g_quiet = args.CheckCmdLineFlag("quiet");
	args.GetCmdLineArgument("mtx", mtx_filename);
	args.GetCmdLineArgument("output", output_csv);
	args.GetCmdLineArgument("threads", g_omp_threads);
	args.GetCmdLineArgument("max_iters", max_iters);
	args.GetCmdLineArgument("tolerance", tolerance);
	args.GetCmdLineArgument("timing_iters", timing_iterations);

	// Check if matrix is specified
	if (mtx_filename.empty())
	{
		fprintf(stderr, "Usage: %s --mtx=<filename> [options]\n", argv[0]);
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  --output=<csv_file>     Output CSV file (default: data/gflops/<matrix>_gflops.csv)\n");
		fprintf(stderr, "  --timing_iters=<N>      Number of timing iterations (default: 3)\n");
		fprintf(stderr, "  --threads=<N>           Number of OpenMP threads\n");
		fprintf(stderr, "  --max_iters=<N>         Max CG iterations (default: 100000)\n");
		fprintf(stderr, "  --tolerance=<X>         Convergence tolerance (default: 1e-5)\n");
		fprintf(stderr, "  --quiet                 Suppress verbose output\n");
		exit(1);
	}

	// Set number of threads
	if (g_omp_threads != -1)
	{
		omp_set_num_threads(g_omp_threads);
	}

	// Generate output filename if not specified
	if (output_csv.empty())
	{
		std::string matrix_name = GetMatrixBaseName(mtx_filename);
		output_csv = "data/gflops/" + matrix_name + "_gflops.csv";
	}

	// Run benchmarks
	std::vector<BenchmarkResult> results;
	RunAllBenchmarks<long double, int>(mtx_filename, max_iters, tolerance, timing_iterations, results);

	// Save results
	SaveResultsToCSV(output_csv, results);

	if (!g_quiet)
		printf("All benchmarks completed.\n");

	return 0;
}
