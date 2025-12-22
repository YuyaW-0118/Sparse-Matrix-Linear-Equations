/******************************************************************************
 * Matrix Statistics Tool
 *
 * Calculates per-row nonzero count statistics for all .mtx files in a directory.
 * Outputs: filename, num_rows, num_cols, num_nonzeros, row_length_mean, row_length_std_dev
 ******************************************************************************/

#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>

#include "../sparse_matrix.h"

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    std::string mtx_dir = "../download/final_mtx2";

    if (argc > 1)
    {
        mtx_dir = argv[1];
    }

    // Collect all .mtx files
    std::vector<std::string> mtx_files;
    for (const auto &entry : fs::directory_iterator(mtx_dir))
    {
        if (entry.path().extension() == ".mtx")
        {
            mtx_files.push_back(entry.path().string());
        }
    }

    // Sort files alphabetically
    std::sort(mtx_files.begin(), mtx_files.end());

    // Print CSV header
    std::cout << "filename,num_rows,num_cols,num_nonzeros,row_length_mean,row_length_std_dev" << std::endl;

    // Process each file
    for (const auto &mtx_path : mtx_files)
    {
        // Extract filename
        std::string filename = fs::path(mtx_path).filename().string();

        // Load COO matrix
        CooMatrix<double, int> coo_matrix;
        coo_matrix.InitMarket(mtx_path, 1.0, false);

        // Convert to CSR
        CsrMatrix<double, int> csr_matrix;
        csr_matrix.Init(coo_matrix, false);

        // Get statistics
        GraphStats stats = csr_matrix.Stats();

        // Output CSV row
        std::cout << filename << ","
                  << stats.num_rows << ","
                  << stats.num_cols << ","
                  << stats.num_nonzeros << ","
                  << stats.row_length_mean << ","
                  << stats.row_length_std_dev << std::endl;
    }

    return 0;
}
