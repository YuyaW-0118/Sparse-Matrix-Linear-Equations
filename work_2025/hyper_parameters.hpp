#ifndef HYPER_PARAMETERS_HPP
#define HYPER_PARAMETERS_HPP

//---------------------------------------------------------------------
// Globals, constants, and type declarations
//---------------------------------------------------------------------

bool g_quiet = false;	 // Whether to display stats in CSV format
bool g_verbose = false;	 // Whether to display output to console
bool g_verbose2 = false; // Whether to display input to console
int g_omp_threads = 10;	 // Number of openMP threads
int g_expected_calls = 1000000;
bool g_input_row_major = false;
bool g_output_row_major = false;

#endif // HYPER_PARAMETERS_HPP
