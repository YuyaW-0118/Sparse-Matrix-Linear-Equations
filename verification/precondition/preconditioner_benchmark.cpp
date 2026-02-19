/******************************************************************************
 * preconditioner_benchmark.cpp
 *
 * Compare preconditioner performance for CG solver.
 *
 * Parameters:
 *   - num_vectors = 32 (fixed)
 *   - threads = 16 (fixed)
 *   - timing_iterations = 5
 *
 * Preconditioners tested:
 *   - NONE: No preconditioning (standard CG)
 *   - IC0: Incomplete Cholesky IC(0)
 *   - SPAI: Sparse Approximate Inverse
 *
 * Output per matrix: data/prepare/{mtx_name}_prepare.csv
 *   Columns: PREPARE_TYPE, preprocess_ms, solve_ms, total_ms, gflops, iterations
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
#include <iomanip>
#include <numeric>
#include <cmath>
#include <filesystem>

#include <mkl.h>

#include "../../sparse_matrix.h"
#include "../../utils.h"
#include "../../work_2025/hyper_parameters.hpp"
#include "../../work_2025/main/no_pretreatment.hpp"
#include "../../work_2025/main/incomplete_cholesky.hpp"
#include "../../work_2025/main/sparse_approximate_inverse.hpp"

namespace fs = std::filesystem;

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
	std::string prepare_type;
	double preprocess_ms;
	double solve_ms;
	double total_ms;
	double gflops;
	int iterations;
};

/**
 * Run CG without preconditioning
 */
template <typename ValueT, typename OffsetT>
BenchmarkResult RunNoPreconditioner(
	CsrMatrix<ValueT, OffsetT> &csr_matrix,
	ValueT *b_vectors,
	ValueT *x_solutions,
	int num_vectors,
	int max_iters,
	ValueT tolerance,
	int timing_iterations,
	SpmmKernel kernel_type)
{
	BenchmarkResult result;
	result.prepare_type = "NONE";
	result.preprocess_ms = 0.0;

	double flops_per_iter = (2.0 * csr_matrix.num_nonzeros + 10.0 * csr_matrix.num_rows) * num_vectors;

	double min_ms = std::numeric_limits<double>::max();
	int best_iters = 0;

	for (int t = 0; t < timing_iterations; ++t)
	{
		// Reset solutions
		std::fill(x_solutions, x_solutions + csr_matrix.num_rows * num_vectors, ValueT(0));

		CpuTimer timer;
		timer.Start();
		int iters = CGSolveMultiple(csr_matrix, b_vectors, x_solutions, num_vectors, max_iters, tolerance, kernel_type);
		timer.Stop();

		double elapsed = timer.ElapsedMillis();
		if (elapsed < min_ms)
		{
			min_ms = elapsed;
			best_iters = iters;
		}
	}

	result.solve_ms = min_ms;
	result.total_ms = min_ms;
	result.iterations = best_iters;
	result.gflops = (flops_per_iter * best_iters) / (min_ms / 1000.0) / 1e9;

	return result;
}

/**
 * Run PCG with IC(0) preconditioning
 */
template <typename ValueT, typename OffsetT>
BenchmarkResult RunIC0Preconditioner(
	CsrMatrix<ValueT, OffsetT> &csr_matrix,
	ValueT *b_vectors,
	ValueT *x_solutions,
	int num_vectors,
	int max_iters,
	ValueT tolerance,
	int timing_iterations,
	SpmmKernel kernel_type)
{
	BenchmarkResult result;
	result.prepare_type = "IC0";

	// Perform IC factorization (only once)
	CsrMatrix<ValueT, OffsetT> L_matrix;
	CpuTimer ic_timer;
	ic_timer.Start();
	bool ic_success = IncompleteCholesky(csr_matrix, L_matrix);
	ic_timer.Stop();

	if (!ic_success)
	{
		result.preprocess_ms = -1.0;
		result.solve_ms = -1.0;
		result.total_ms = -1.0;
		result.gflops = 0.0;
		result.iterations = -1;
		return result;
	}

	result.preprocess_ms = ic_timer.ElapsedMillis();

	// Compute L transpose
	CsrMatrix<ValueT, OffsetT> L_transpose;
	TransposeCsr(L_matrix, L_transpose);

	// FLOPS for PCG includes preconditioner solves
	double nnz_l = L_matrix.num_nonzeros;
	double flops_per_iter = (2.0 * csr_matrix.num_nonzeros + 4.0 * nnz_l + 12.0 * csr_matrix.num_rows) * num_vectors;

	double min_ms = std::numeric_limits<double>::max();
	int best_iters = 0;

	for (int t = 0; t < timing_iterations; ++t)
	{
		// Reset solutions
		std::fill(x_solutions, x_solutions + csr_matrix.num_rows * num_vectors, ValueT(0));

		CpuTimer timer;
		timer.Start();
		int iters = PCGSolveMultiple(csr_matrix, L_matrix, L_transpose, b_vectors, x_solutions,
									 num_vectors, max_iters, tolerance, kernel_type);
		timer.Stop();

		double elapsed = timer.ElapsedMillis();
		if (elapsed < min_ms)
		{
			min_ms = elapsed;
			best_iters = iters;
		}
	}

	result.solve_ms = min_ms;
	result.total_ms = result.preprocess_ms + min_ms;
	result.iterations = best_iters;
	result.gflops = (flops_per_iter * best_iters) / (min_ms / 1000.0) / 1e9;

	L_matrix.Clear();
	L_transpose.Clear();

	return result;
}

/**
 * Run PCG with SPAI preconditioning
 */
template <typename ValueT, typename OffsetT>
BenchmarkResult RunSPAIPreconditioner(
	CsrMatrix<ValueT, OffsetT> &csr_matrix,
	ValueT *b_vectors,
	ValueT *x_solutions,
	int num_vectors,
	int max_iters,
	ValueT tolerance,
	int timing_iterations,
	SpmmKernel kernel_type)
{
	BenchmarkResult result;
	result.prepare_type = "SPAI";

	// Perform SPAI factorization (only once)
	printf("    Performing SPAI factorization (rows=%d, nnz=%d)...\n",
		   csr_matrix.num_rows, csr_matrix.num_nonzeros);
	fflush(stdout);
	CsrMatrix<ValueT, OffsetT> spai_matrix;
	CpuTimer spai_timer;
	spai_timer.Start();
	bool spai_success = SparseApproximateInversion(csr_matrix, spai_matrix);
	spai_timer.Stop();
	printf("    SPAI factorization completed: success=%d, time=%.3f ms\n",
		   spai_success, spai_timer.ElapsedMillis());
	fflush(stdout);

	if (!spai_success)
	{
		result.preprocess_ms = -1.0;
		result.solve_ms = -1.0;
		result.total_ms = -1.0;
		result.gflops = 0.0;
		result.iterations = -1;
		return result;
	}

	result.preprocess_ms = spai_timer.ElapsedMillis();

	// FLOPS for SPAI-preconditioned CG
	double flops_per_iter = (4.0 * csr_matrix.num_nonzeros + 12.0 * csr_matrix.num_rows) * num_vectors;

	double min_ms = std::numeric_limits<double>::max();
	int best_iters = 0;

	for (int t = 0; t < timing_iterations; ++t)
	{
		// Reset solutions
		std::fill(x_solutions, x_solutions + csr_matrix.num_rows * num_vectors, ValueT(0));

		CpuTimer timer;
		timer.Start();
		int iters = SPAISolveMultiple(csr_matrix, spai_matrix, b_vectors, x_solutions,
									  num_vectors, max_iters, tolerance, kernel_type);
		timer.Stop();

		double elapsed = timer.ElapsedMillis();
		if (elapsed < min_ms)
		{
			min_ms = elapsed;
			best_iters = iters;
		}
	}

	result.solve_ms = min_ms;
	result.total_ms = result.preprocess_ms + min_ms;
	result.iterations = best_iters;
	result.gflops = (flops_per_iter * best_iters) / (min_ms / 1000.0) / 1e9;

	return result;
}

/**
 * Save results to CSV
 */
void SaveResultsToCSV(
	const std::string &filename,
	const std::vector<BenchmarkResult> &results)
{
	std::ofstream ofs(filename);
	if (!ofs.is_open())
	{
		fprintf(stderr, "Error: Cannot open %s for writing\n", filename.c_str());
		return;
	}

	ofs << "PREPARE_TYPE,preprocess_ms,solve_ms,total_ms,gflops,iterations" << std::endl;
	for (const auto &r : results)
	{
		ofs << r.prepare_type << ","
			<< std::fixed << std::setprecision(3) << r.preprocess_ms << ","
			<< std::fixed << std::setprecision(3) << r.solve_ms << ","
			<< std::fixed << std::setprecision(3) << r.total_ms << ","
			<< std::fixed << std::setprecision(2) << r.gflops << ","
			<< r.iterations << std::endl;
	}

	ofs.close();
	printf("Results saved to: %s\n", filename.c_str());
}

/**
 * Process single matrix
 */
template <typename ValueT, typename OffsetT>
void ProcessMatrix(
	const std::string &mtx_filename,
	const std::string &output_dir,
	int num_vectors,
	int max_iters,
	ValueT tolerance,
	int timing_iterations,
	SpmmKernel kernel_type)
{
	// Load matrix
	CooMatrix<ValueT, OffsetT> coo_matrix;
	coo_matrix.InitMarket(mtx_filename, 1.0, false);

	if ((coo_matrix.num_rows == 1) || (coo_matrix.num_cols == 1) || (coo_matrix.num_nonzeros == 1))
	{
		printf("Skipping trivial matrix: %s\n", mtx_filename.c_str());
		return;
	}

	CsrMatrix<ValueT, OffsetT> csr_matrix(coo_matrix);
	coo_matrix.Clear();

	std::string matrix_name = GetMatrixBaseName(mtx_filename);
	printf("Processing: %s (rows=%d, nnz=%d)\n",
		   matrix_name.c_str(), csr_matrix.num_rows, csr_matrix.num_nonzeros);

	// Allocate vectors
	ValueT *b_vectors = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows * num_vectors, 4096);
	ValueT *x_solutions = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows * num_vectors, 4096);

	// Initialize RHS vectors
	srand(42);
	for (long long i = 0; i < (long long)csr_matrix.num_rows * num_vectors; ++i)
		b_vectors[i] = static_cast<ValueT>(rand()) / static_cast<ValueT>(RAND_MAX);

	std::vector<BenchmarkResult> results;

	// Test 1: No preconditioning
	printf("  Testing NONE...\n");
	BenchmarkResult none_result = RunNoPreconditioner(
		csr_matrix, b_vectors, x_solutions, num_vectors,
		max_iters, tolerance, timing_iterations, kernel_type);
	results.push_back(none_result);
	printf("    solve_ms=%.3f, gflops=%.2f, iters=%d\n",
		   none_result.solve_ms, none_result.gflops, none_result.iterations);

	// Test 2: IC(0) preconditioning
	printf("  Testing IC0...\n");
	BenchmarkResult ic0_result = RunIC0Preconditioner(
		csr_matrix, b_vectors, x_solutions, num_vectors,
		max_iters, tolerance, timing_iterations, kernel_type);
	results.push_back(ic0_result);
	if (ic0_result.iterations >= 0)
	{
		printf("    preprocess_ms=%.3f, solve_ms=%.3f, gflops=%.2f, iters=%d\n",
			   ic0_result.preprocess_ms, ic0_result.solve_ms, ic0_result.gflops, ic0_result.iterations);
	}
	else
	{
		printf("    IC0 factorization failed\n");
	}

	// Test 3: SPAI preconditioning
	printf("  Testing SPAI...\n");
	BenchmarkResult spai_result = RunSPAIPreconditioner(
		csr_matrix, b_vectors, x_solutions, num_vectors,
		max_iters, tolerance, timing_iterations, kernel_type);
	results.push_back(spai_result);
	if (spai_result.iterations >= 0)
	{
		printf("    preprocess_ms=%.3f, solve_ms=%.3f, gflops=%.2f, iters=%d\n",
			   spai_result.preprocess_ms, spai_result.solve_ms, spai_result.gflops, spai_result.iterations);
	}
	else
	{
		printf("    SPAI factorization failed\n");
	}

	// Save results
	std::string output_file = output_dir + "/" + matrix_name + "_prepare.csv";
	SaveResultsToCSV(output_file, results);

	// Cleanup
	mkl_free(b_vectors);
	mkl_free(x_solutions);
}

/**
 * Main
 */
int main(int argc, char **argv)
{
	// Fixed parameters
	int num_vectors = 32;
	int num_threads = 16;
	int timing_iterations = 5;
	int max_iters = 100000;
	double tolerance = 1.0e-5;
	SpmmKernel kernel_type = MERGE;

	// Configurable parameters
	std::string mtx_dir = "../download/final_mtx";
	std::string output_dir = "../data/prepare";

	// Parse command line arguments
	for (int i = 1; i < argc; ++i)
	{
		std::string arg = argv[i];
		if (arg.find("--mtx_dir=") == 0)
		{
			mtx_dir = arg.substr(10);
		}
		else if (arg.find("--output_dir=") == 0)
		{
			output_dir = arg.substr(13);
		}
		else if (arg.find("--num_vectors=") == 0)
		{
			num_vectors = std::stoi(arg.substr(14));
		}
		else if (arg.find("--threads=") == 0)
		{
			num_threads = std::stoi(arg.substr(10));
		}
		else if (arg.find("--timing_iters=") == 0)
		{
			timing_iterations = std::stoi(arg.substr(15));
		}
	}

	// Set thread count
	omp_set_num_threads(num_threads);
	g_omp_threads = num_threads;
	g_quiet = true;

	// Create output directory
	fs::create_directories(output_dir);

	// Find all .mtx files
	std::vector<std::string> mtx_files;
	for (const auto &entry : fs::directory_iterator(mtx_dir))
	{
		if (entry.path().extension() == ".mtx")
		{
			mtx_files.push_back(entry.path().string());
		}
	}
	std::sort(mtx_files.begin(), mtx_files.end());

	if (mtx_files.empty())
	{
		fprintf(stderr, "Error: No .mtx files found in %s\n", mtx_dir.c_str());
		return 1;
	}

	printf("=== Preconditioner Benchmark ===\n");
	printf("Matrix directory: %s\n", mtx_dir.c_str());
	printf("Output directory: %s\n", output_dir.c_str());
	printf("Number of matrices: %zu\n", mtx_files.size());
	printf("num_vectors: %d\n", num_vectors);
	printf("threads: %d\n", num_threads);
	printf("timing_iterations: %d\n", timing_iterations);
	printf("\n");

	// Process all matrices
	for (const auto &mtx_file : mtx_files)
	{
		ProcessMatrix<double, int>(
			mtx_file, output_dir, num_vectors,
			max_iters, tolerance, timing_iterations, kernel_type);
		printf("\n");
	}

	printf("All benchmarks completed.\n");

	return 0;
}
