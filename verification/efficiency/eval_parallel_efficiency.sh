#!/bin/bash
#
# eval_parallel_efficiency.sh
#
# Measure parallel speedup and efficiency for CG solver.
#
# Speedup = T(1) / T(n)
# Efficiency = Speedup / n
#
# Tests thread counts: 1, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18
# Fixed num_vectors = 16
#


set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"


# Parameters
MTX_DIR="${PROJECT_DIR}/download/final_mtx"
OUTPUT_DIR="${PROJECT_DIR}/data/parallel"
TIMING_ITERS=${TIMING_ITERS:-3}
NUM_VECTORS=${NUM_VECTORS:-16}

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build the tool
echo "Building parallel_efficiency..."
make -C "${SCRIPT_DIR}" parallel_efficiency

# Check if executable exists
DRIVER="${SCRIPT_DIR}/_parallel_efficiency"
if [ ! -x "${DRIVER}" ]; then
    echo "Error: ${DRIVER} not found or not executable"
    exit 1
fi

echo ""
echo "=== Parallel Efficiency Benchmark ==="
echo "Matrix directory: ${MTX_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "num_vectors: ${NUM_VECTORS}"
echo "Timing iterations: ${TIMING_ITERS}"
echo ""

# Run benchmark
"${DRIVER}" \
    --mtx_dir="${MTX_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --num_vectors="${NUM_VECTORS}" \
    --timing_iters="${TIMING_ITERS}"

echo ""
echo "=== Results ==="
echo ""

# Display summary CSV
if [ -f "${OUTPUT_DIR}/parallel_efficiency.csv" ]; then
    echo "Summary (${OUTPUT_DIR}/parallel_efficiency.csv):"
    cat "${OUTPUT_DIR}/parallel_efficiency.csv"
    echo ""
fi

echo "Generated files:"
ls -la "${OUTPUT_DIR}"/*.csv 2>/dev/null || echo "  (no files found)"
