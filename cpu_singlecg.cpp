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
#include <string>
#include <cmath>

#include <mkl.h>

#include "sparse_matrix.h"
#include "utils.h"
#include "work_2025/hyper_parameters.hpp"
#include "work_2025/main/single_strategy.hpp"

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
 * Structure to hold benchmark results
 */
struct BenchmarkResult
{
	std::string matrix_name;
	std::string kernel_name; // Will be "SINGLE_LOOP"
	int num_vectors;
	double min_ms;
	double gflops;
	long long iterations; // Total iterations across all vectors
};

/**
 * Run CG benchmark for the Single Strategy
 */
template <typename ValueT, typename OffsetT>
BenchmarkResult RunCGSimpleBenchmark(
	CsrMatrix<ValueT, OffsetT> &csr_matrix,
	const std::string &matrix_name,
	int num_vectors,
	int max_iters,
	ValueT tolerance,
	int timing_iterations)
{
	BenchmarkResult result;
	result.matrix_name = matrix_name;
	result.kernel_name = "SINGLE_LOOP";
	result.num_vectors = num_vectors;

	// Allocate vectors (Using MKL malloc for alignment)
	// Flattened arrays: [num_vectors * num_rows]
	ValueT *b_vectors = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows * num_vectors, 4096);
	ValueT *x_solutions = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows * num_vectors, 4096);

	// Initialize RHS vectors with random values
	// Note: Using same seed logic as cpu_multicg2 for consistency
	srand(42);
	for (long long i = 0; i < (long long)csr_matrix.num_rows * num_vectors; ++i)
		b_vectors[i] = static_cast<ValueT>(rand()) / static_cast<ValueT>(RAND_MAX);

	double threshold = calculate_threshold(b_vectors, csr_matrix.num_rows, tolerance);

	double flops_per_iter_single = 2.0 * csr_matrix.num_nonzeros + 10.0 * csr_matrix.num_rows;

	// Run benchmark
	double min_ms = 0.0;
	double total_iters_double = 0.0;

	// Call the Single Strategy Solver
	TestCGSolveSingle(csr_matrix, b_vectors, x_solutions, max_iters, threshold, num_vectors,
					  timing_iterations, min_ms, total_iters_double);

	result.min_ms = min_ms;
	result.iterations = static_cast<long long>(total_iters_double);

	// GFLOPS = (FLOPS_PER_ITER * TOTAL_ITERS) / (TIME_SEC * 1e9)
	result.gflops = (flops_per_iter_single * total_iters_double) / (min_ms / 1000.0) / 1e9;

	// Cleanup
	mkl_free(b_vectors);
	mkl_free(x_solutions);

	return result;
}

/**
 * Run all benchmarks for a matrix
 */
template <typename ValueT, typename OffsetT>
void RunAllSimpleBenchmarks(
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

	// Test configurations (Same vector counts as cpu_multicg2)
	// std::vector<int> num_vectors_list = {2, 4, 8, 16, 32, 64, 128};
	std::vector<int> num_vectors_list = {16};

	// Run benchmarks
	for (int num_vectors : num_vectors_list)
	{
		if (!g_quiet)
		{
			printf("  Testing: num_vectors=%d, method=SINGLE_LOOP\n", num_vectors);
		}

		BenchmarkResult result = RunCGSimpleBenchmark(
			csr_matrix, matrix_name, num_vectors, max_iters, tolerance,
			timing_iterations);

		results.push_back(result);

		printf("    %s, L=%d, method=SINGLE_LOOP: %.3f ms, %lld iters, %.2f GFLOPS\n",
			   matrix_name.c_str(), num_vectors,
			   result.min_ms, result.iterations, result.gflops);
	}

	// csr_matrix.Clear();
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
		// Try to create directory if it fails? (Standard C++ fstream won't create dirs)
		// For now, assume the user/script creates 'data/simple_gflops/'
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
	int timing_iterations = 1;

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
		fprintf(stderr, "  --output=<csv_file>     Output CSV file (default: data/simple_gflops/<matrix>_gflops.csv)\n");
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
		// Note: The prompt requested "data/simple_gflops/"
		output_csv = "data/simple_gflops/" + matrix_name + "_gflops.csv";
	}

	// Run benchmarks
	std::vector<BenchmarkResult> results;
	RunAllSimpleBenchmarks<double, int>(mtx_filename, max_iters, tolerance, timing_iterations, results);

	// Save results
	SaveResultsToCSV(output_csv, results);

	if (!g_quiet)
		printf("All simple benchmarks completed.\n");

	return 0;
}