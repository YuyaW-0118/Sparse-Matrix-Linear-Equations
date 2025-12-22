#!/bin/bash
#
# eval_gflops.sh
#
# Benchmark CG solver with 3 SpMM methods across multiple num_vectors values
# for all .mtx files in download/final_mtx/
#
# Output: CSV files in data/gflops/
#


set -e


# Directory settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MTX_DIR="${SCRIPT_DIR}/download/final_mtx"
OUTPUT_DIR="${SCRIPT_DIR}/data/gflops"
DRIVER="${SCRIPT_DIR}/_cpu_multicg2_driver"

# Default parameters
THREADS=${THREADS:-8}
TIMING_ITERS=${TIMING_ITERS:-3}

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build the driver if needed
echo "Building cpu_multicg2..."
make -C "${SCRIPT_DIR}" cpu_multicg2

# Check if driver exists
if [ ! -x "${DRIVER}" ]; then
    echo "Error: Driver ${DRIVER} not found or not executable"
    exit 1
fi

# Find all .mtx files
MTX_FILES=$(find "${MTX_DIR}" -name "*.mtx" -type f | sort)

if [ -z "${MTX_FILES}" ]; then
    echo "Error: No .mtx files found in ${MTX_DIR}"
    exit 1
fi

echo "=== GFLOPS Benchmark ==="
echo "Threads: ${THREADS}"
echo "Timing iterations: ${TIMING_ITERS}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Process each matrix file
for mtx_file in ${MTX_FILES}; do
    mtx_name=$(basename "${mtx_file}" .mtx)
    if [ "${mtx_name}" != "msc00726" ]; then
        continue
    fi
    output_csv="${OUTPUT_DIR}/${mtx_name}_gflops.csv"

    echo "Processing: ${mtx_name}"
    echo "  Input: ${mtx_file}"
    echo "  Output: ${output_csv}"

    "${DRIVER}" \
        --mtx="${mtx_file}" \
        --output="${output_csv}" \
        --threads="${THREADS}" \
        --timing_iters="${TIMING_ITERS}" \
        --quiet

    echo "  Done."
    echo ""
done

echo "=== All benchmarks completed ==="
echo "Results saved to: ${OUTPUT_DIR}"

# List generated files
echo ""
echo "Generated files:"
ls -la "${OUTPUT_DIR}"/*.csv 2>/dev/null || echo "  (no CSV files found)"
