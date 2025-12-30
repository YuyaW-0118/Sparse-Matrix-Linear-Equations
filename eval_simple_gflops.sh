#!/bin/bash

# -----------------------------------------------------------------------------
# eval_simple_gflops.sh
# 
# compile and run cpu_singlecg benchmark for multiple matrices
# -----------------------------------------------------------------------------

set -e

# Hyper parameters (Match eval_gflops.sh defaults)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MTX_DIR="${SCRIPT_DIR}/download/"
OUTPUT_DIR="${SCRIPT_DIR}/data/simple_gflops"
DRIVER="${SCRIPT_DIR}/cpu_singlecg"


THREADS=${THREADS:-8}
TIMING_ITERS=${TIMING_ITERS:-1}

mkdir -p "$OUTPUT_DIR"

# -----------------------------------------------------------------------------
# 1. Compile
# -----------------------------------------------------------------------------
echo "Compiling cpu_singlecg..."
make -C "$SCRIPT_DIR" cpu_singlecg

if [ ! -x "${DRIVER}" ]; then
	echo "Error: Driver ${DRIVER} not found or not executable"
	exit 1
fi


# -----------------------------------------------------------------------------
# 3. Benchmark Loop
# -----------------------------------------------------------------------------

MTX_FILES =$(find "${MTX_DIR}" -name "*.mtx" -type f | sort)

if [ -z "${MTX_FILES}" ]; then
	echo "Error: No .mtx files found in ${MTX_DIR}"
	exit 1
fi

echo "=== Simple GFLOPS Benchmark ==="
echo "Threads: $THREADS"
echo "Timing iterations: $TIMING_ITERS"
echo "Output directory: $OUTPUT_DIR"
echo ""

for mtx_file in ${MTX_FILES}; do
	mtx_name=$(basename "${mtx_file}" .mtx)
	output_csv="${OUTPUT_DIR}/${mtx_name}_gflops.csv"

	echo "Processing: $mtx_name"

	# Run the benchmark
	"${DRIVER}" \
		--mtx="$mtx_file" \
		--output_csv="$output_csv" \
		--threads="$THREADS" \
		--timing_iters="$TIMING_ITERS" \
		--quiet

	echo "Completed: $mtx_name"
	echo "---"
done

echo "=== All benchmarks completed ==="
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
ls -la "${OUTPUT_DIR}"/*.csv 2>/dev/null || echo "  (no CSV files found)"
