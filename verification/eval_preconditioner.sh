#!/bin/bash
#
# eval_preconditioner.sh
#
# Compare preconditioner performance for CG solver.
#
# Parameters (fixed):
#   - num_vectors = 32
#   - threads = 16
#   - timing_iterations = 5
#
# Preconditioners tested:
#   - NONE: No preconditioning
#   - IC0: Incomplete Cholesky IC(0)
#   - SPAI: Sparse Approximate Inverse
#
# Output: data/prepare/{mtx_name}_prepare.csv
#


set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"


# Fixed parameters
NUM_VECTORS=2
THREADS=8
TIMING_ITERS=1

# Configurable directories
MTX_DIR="${PROJECT_DIR}/download/final_mtx"
OUTPUT_DIR="${PROJECT_DIR}/data/prepare"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build the tool
echo "Building preconditioner_benchmark..."
make -C "${SCRIPT_DIR}" preconditioner_benchmark

# Check if executable exists
DRIVER="${SCRIPT_DIR}/_preconditioner_benchmark"
if [ ! -x "${DRIVER}" ]; then
    echo "Error: ${DRIVER} not found or not executable"
    exit 1
fi

echo ""
echo "=== Preconditioner Benchmark ==="
echo "Matrix directory: ${MTX_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "num_vectors: ${NUM_VECTORS}"
echo "threads: ${THREADS}"
echo "timing_iterations: ${TIMING_ITERS}"
echo ""

# Run benchmark
"${DRIVER}" \
    --mtx_dir="${MTX_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --num_vectors="${NUM_VECTORS}" \
    --threads="${THREADS}" \
    --timing_iters="${TIMING_ITERS}"

echo ""
echo "=== Results ==="
echo ""

# Display generated files
echo "Generated files:"
ls -la "${OUTPUT_DIR}"/*.csv 2>/dev/null || echo "  (no files found)"
