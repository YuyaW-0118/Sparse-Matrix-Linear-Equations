/******************************************************************************
 * parallel_efficiency.cpp
 *
 * Measure parallel speedup and efficiency for CG solver.
 *
 * Speedup = T(1) / T(n)  where T(n) is execution time with n threads
 * Efficiency = Speedup / n
 *
 * Tests thread counts: 1, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18
 * Fixed num_vectors = 16
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

#include "../sparse_matrix.h"
#include "../utils.h"
#include "../work_2025/hyper_parameters.hpp"
#include "../work_2025/main/no_pretreatment.hpp"

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
 * Structure to hold timing results
 */
struct TimingResult
{
	std::string matrix_name;
	int num_threads;
	double time_ms;
	double gflops;
	int iterations;
};

/**
 * Run CG benchmark with specific thread count
 */
template <typename ValueT, typename OffsetT>
TimingResult RunBenchmark(
	CsrMatrix<ValueT, OffsetT> &csr_matrix,
	const std::string &matrix_name,
	int num_threads,
	int num_vectors,
	int max_iters,
	ValueT tolerance,
	int timing_iterations,
	SpmmKernel kernel_type)
{
	TimingResult result;
	result.matrix_name = matrix_name;
	result.num_threads = num_threads;

	// Set thread count
	omp_set_num_threads(num_threads);
	g_omp_threads = num_threads;

	// Allocate vectors
	ValueT *b_vectors = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows * num_vectors, 4096);
	ValueT *x_solutions = (ValueT *)mkl_malloc(sizeof(ValueT) * csr_matrix.num_rows * num_vectors, 4096);

	// Initialize RHS vectors
	srand(42);
	for (long long i = 0; i < (long long)csr_matrix.num_rows * num_vectors; ++i)
		b_vectors[i] = static_cast<ValueT>(rand()) / static_cast<ValueT>(RAND_MAX);

	// Calculate FLOPS
	double flops_per_iter_single = 2.0 * csr_matrix.num_nonzeros + 10.0 * csr_matrix.num_rows;
	double flops_per_iter_multi = flops_per_iter_single * num_vectors;

	// Run benchmark
	double min_ms;
	double iters_of_min_ms;
	TestCGMultipleRHS(csr_matrix, b_vectors, x_solutions, max_iters, tolerance, num_vectors,
					  timing_iterations, kernel_type, min_ms, iters_of_min_ms);

	result.time_ms = min_ms;
	result.iterations = static_cast<int>(iters_of_min_ms);
	result.gflops = (flops_per_iter_multi * iters_of_min_ms) / (min_ms / 1000.0) / 1e9;

	mkl_free(b_vectors);
	mkl_free(x_solutions);

	return result;
}

/**
 * Process single matrix and return timing results for all thread counts
 */
template <typename ValueT, typename OffsetT>
std::vector<TimingResult> ProcessMatrix(
	const std::string &mtx_filename,
	const std::vector<int> &thread_counts,
	int num_vectors,
	int max_iters,
	ValueT tolerance,
	int timing_iterations,
	SpmmKernel kernel_type)
{
	std::vector<TimingResult> results;

	// Load matrix
	CooMatrix<ValueT, OffsetT> coo_matrix;
	coo_matrix.InitMarket(mtx_filename, 1.0, true);

	if ((coo_matrix.num_rows == 1) || (coo_matrix.num_cols == 1) || (coo_matrix.num_nonzeros == 1))
	{
		printf("Skipping trivial matrix: %s\n", mtx_filename.c_str());
		return results;
	}

	CsrMatrix<ValueT, OffsetT> csr_matrix(coo_matrix);
	coo_matrix.Clear();

	std::string matrix_name = GetMatrixBaseName(mtx_filename);
	printf("Processing: %s (rows=%d, nnz=%d)\n",
		   matrix_name.c_str(), csr_matrix.num_rows, csr_matrix.num_nonzeros);

	// Run for each thread count
	for (int num_threads : thread_counts)
	{
		printf("  Threads=%2d: ", num_threads);
		fflush(stdout);

		TimingResult result = RunBenchmark(
			csr_matrix, matrix_name, num_threads, num_vectors,
			max_iters, tolerance, timing_iterations, kernel_type);

		results.push_back(result);
		printf("%.3f ms, %.2f GFLOPS\n", result.time_ms, result.gflops);
	}

	csr_matrix.Clear();
	return results;
}

/**
 * Calculate speedup and efficiency
 */
struct EfficiencyResult
{
	int num_threads;
	double avg_time_ms;
	double avg_gflops;
	double speedup;
	double efficiency;
};

std::vector<EfficiencyResult> CalculateEfficiency(
	const std::vector<std::vector<TimingResult>> &all_results,
	const std::vector<int> &thread_counts)
{
	std::vector<EfficiencyResult> efficiency_results;

	// For each thread count, calculate average across all matrices
	for (size_t t = 0; t < thread_counts.size(); ++t)
	{
		int num_threads = thread_counts[t];
		std::vector<double> times;
		std::vector<double> gflops_list;

		for (const auto &matrix_results : all_results)
		{
			if (t < matrix_results.size())
			{
				times.push_back(matrix_results[t].time_ms);
				gflops_list.push_back(matrix_results[t].gflops);
			}
		}

		if (times.empty())
			continue;

		double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
		double avg_gflops = std::accumulate(gflops_list.begin(), gflops_list.end(), 0.0) / gflops_list.size();

		EfficiencyResult result;
		result.num_threads = num_threads;
		result.avg_time_ms = avg_time;
		result.avg_gflops = avg_gflops;

		efficiency_results.push_back(result);
	}

	// Calculate speedup and efficiency (based on single-thread performance)
	if (!efficiency_results.empty())
	{
		double base_time = efficiency_results[0].avg_time_ms; // T(1)
		double base_gflops = efficiency_results[0].avg_gflops;

		for (auto &r : efficiency_results)
		{
			// Speedup based on time: T(1) / T(n)
			r.speedup = base_time / r.avg_time_ms;
			// Efficiency = Speedup / n
			r.efficiency = r.speedup / r.num_threads;
		}
	}

	return efficiency_results;
}

/**
 * Save results to CSV
 */
void SaveResultsToCSV(
	const std::string &filename,
	const std::vector<EfficiencyResult> &results)
{
	std::ofstream ofs(filename);
	if (!ofs.is_open())
	{
		fprintf(stderr, "Error: Cannot open %s for writing\n", filename.c_str());
		return;
	}

	ofs << "num_threads,avg_time_ms,avg_gflops,speedup,efficiency" << std::endl;
	for (const auto &r : results)
	{
		ofs << r.num_threads << ","
			<< std::fixed << std::setprecision(3) << r.avg_time_ms << ","
			<< std::fixed << std::setprecision(2) << r.avg_gflops << ","
			<< std::fixed << std::setprecision(3) << r.speedup << ","
			<< std::fixed << std::setprecision(4) << r.efficiency << std::endl;
	}

	ofs.close();
	printf("Results saved to: %s\n", filename.c_str());
}

/**
 * Save detailed per-matrix results to CSV
 */
void SaveDetailedResultsToCSV(
	const std::string &filename,
	const std::vector<std::vector<TimingResult>> &all_results)
{
	std::ofstream ofs(filename);
	if (!ofs.is_open())
	{
		fprintf(stderr, "Error: Cannot open %s for writing\n", filename.c_str());
		return;
	}

	ofs << "matrix_name,num_threads,time_ms,gflops,iterations" << std::endl;
	for (const auto &matrix_results : all_results)
	{
		for (const auto &r : matrix_results)
		{
			ofs << r.matrix_name << ","
				<< r.num_threads << ","
				<< std::fixed << std::setprecision(3) << r.time_ms << ","
				<< std::fixed << std::setprecision(2) << r.gflops << ","
				<< r.iterations << std::endl;
		}
	}

	ofs.close();
	printf("Detailed results saved to: %s\n", filename.c_str());
}

/**
 * Main
 */
int main(int argc, char **argv)
{
	// Parameters
	std::string mtx_dir = "../download/final_mtx";
	std::string output_dir = "../data/parallel";
	int num_vectors = 16;
	int max_iters = 100000;
	double tolerance = 1.0e-5;
	int timing_iterations = 3;
	SpmmKernel kernel_type = NONZERO_SPLIT;

	// Thread counts to test
	std::vector<int> thread_counts = {1, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18};

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
		else if (arg.find("--timing_iters=") == 0)
		{
			timing_iterations = std::stoi(arg.substr(15));
		}
	}

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

	printf("=== Parallel Efficiency Benchmark ===\n");
	printf("Matrix directory: %s\n", mtx_dir.c_str());
	printf("Number of matrices: %zu\n", mtx_files.size());
	printf("num_vectors: %d\n", num_vectors);
	printf("Thread counts: ");
	for (int t : thread_counts)
		printf("%d ", t);
	printf("\n\n");

	// Process all matrices
	std::vector<std::vector<TimingResult>> all_results;
	for (const auto &mtx_file : mtx_files)
	{
		auto results = ProcessMatrix<double, int>(
			mtx_file, thread_counts, num_vectors,
			max_iters, tolerance, timing_iterations, kernel_type);

		if (!results.empty())
		{
			all_results.push_back(results);
		}
		printf("\n");
	}

	// Calculate efficiency
	auto efficiency_results = CalculateEfficiency(all_results, thread_counts);

	// Print summary
	printf("\n=== Summary (Average across %zu matrices) ===\n", all_results.size());
	printf("Threads  Time(ms)   GFLOPS   Speedup  Efficiency\n");
	printf("-------  --------   ------   -------  ----------\n");
	for (const auto &r : efficiency_results)
	{
		printf("%7d  %8.3f   %6.2f   %7.3f  %10.4f\n",
			   r.num_threads, r.avg_time_ms, r.avg_gflops, r.speedup, r.efficiency);
	}

	// Save results
	SaveResultsToCSV(output_dir + "/parallel_efficiency.csv", efficiency_results);
	SaveDetailedResultsToCSV(output_dir + "/parallel_efficiency_detailed.csv", all_results);

	printf("\nAll benchmarks completed.\n");

	return 0;
}
