#!/usr/bin/env python3
"""
Plot convergence history of CG solvers.

Usage:
    python plot_errors.py <matrix_name>

Example:
    python plot_errors.py apache1

This script reads CSV files from data/ directory:
    - {matrix_name}_cg_errors.csv
    - {matrix_name}_pcg_ic_errors.csv
    - {matrix_name}_spai_errors.csv

And generates a plot saved to:
    - data/{matrix_name}_plot.png
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_error_data(filepath):
    """Load error data from CSV file if it exists."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Convert pandas Series/ExtensionArray to numpy.ndarray of floats to satisfy matplotlib and static type checks
        iter_arr = df["iteration"].to_numpy()
        err_arr = df["max_error"].to_numpy(dtype=float)
        return iter_arr, err_arr
    # Return (None, None) when file is missing so callers can check existence
    raise FileNotFoundError


def plot_convergence(matrix_name, data_dir="data", fontsize=16):
    """Plot convergence history for all methods."""

    # Define file paths
    cg_file = os.path.join(data_dir, f"{matrix_name}_cg_errors.csv")
    pcg_file = os.path.join(data_dir, f"{matrix_name}_pcg_ic_errors.csv")
    spai_file = os.path.join(data_dir, f"{matrix_name}_spai_errors.csv")

    # Load data
    cg_iter, cg_err = load_error_data(cg_file)
    pcg_iter, pcg_err = load_error_data(pcg_file)
    spai_iter, spai_err = load_error_data(spai_file)

    # Check if at least one file exists
    if cg_err is None and pcg_err is None and spai_err is None:
        print(f"Error: No error data files found for matrix '{matrix_name}'")
        print(f"Expected files in {data_dir}/:")
        print(f"  - {matrix_name}_cg_errors.csv")
        print(f"  - {matrix_name}_pcg_ic_errors.csv")
        print(f"  - {matrix_name}_spai_errors.csv")
        sys.exit(1)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each method if data exists
    if cg_err is not None:
        ax.plot(
            cg_iter,
            cg_err,
            "b-",
            linewidth=1.5,
            label="CG (No Preconditioner)",
            marker="o",
            markevery=max(1, len(cg_iter) // 20),
            markersize=4,
        )

    if pcg_err is not None:
        ax.plot(
            pcg_iter,
            pcg_err,
            "r-",
            linewidth=1.5,
            label="PCG (IC(0))",
            marker="s",
            markevery=max(1, len(pcg_iter) // 20),
            markersize=4,
        )

    if spai_err is not None:
        ax.plot(
            spai_iter,
            spai_err,
            "g-",
            linewidth=1.5,
            label="PCG (SPAI)",
            marker="^",
            markevery=max(1, len(spai_iter) // 20),
            markersize=4,
        )

    # Set logarithmic scale for y-axis
    ax.set_yscale("log")

    # Labels and title
    ax.set_xlabel("Iteration", fontsize=fontsize)
    ax.set_ylabel("Max Relative Error", fontsize=fontsize)
    # ax.set_title(f"Convergence History: {matrix_name} (L = 32)", fontsize=fontsize)

    # Grid and legend
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=fontsize)

    # Tight layout
    plt.tight_layout()

    # Save plot
    output_path = os.path.join(data_dir, f"{matrix_name}_plot.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_errors.py <matrix_name>")
        print("Example: python plot_errors.py apache1")
        sys.exit(1)

    matrix_name = sys.argv[1]

    # Determine data directory (relative to script location or current directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, "data/error_data")

    if not os.path.exists(data_dir):
        # Try current directory
        data_dir = "data"
        if not os.path.exists(data_dir):
            print("Error: data directory not found")
            sys.exit(1)

    plot_convergence(matrix_name, data_dir)


if __name__ == "__main__":
    main()
