#!/bin/bash -l
#SBATCH -J cluster_factor      # Job name
#SBATCH -A g98-2086            # Account (adjust if needed)
#SBATCH -p topola              # Partition
#SBATCH -N 1                   # Number of nodes
#SBATCH --ntasks=1             # Number of tasks
#SBATCH --cpus-per-task=24     # CPUs per task (adjust based on cluster)
#SBATCH --mem=120G             # Memory (5GB per core × 24)
#SBATCH --time=2-00:00:00      # Wall time (2 days)
#SBATCH --output=cluster_factor_%j.log  # Output log (%j = job ID)
#SBATCH --error=cluster_factor_%j.err   # Error log
#SBATCH --hint=nomultithread   # Disable hyperthreading

# =============================================================================
# SLURM Job Script for Cluster Factor Experiment
# =============================================================================
#
# This script runs the cluster factor Monte Carlo experiment on an HPC cluster.
# It loads the full agent.pickle file (no network reduction) and performs
# multiple Monte Carlo runs with comprehensive logging.
#
# Usage:
#   sbatch run_cluster_factor.sl
#
# Monitor job:
#   squeue -u $USER
#   tail -f cluster_factor_<jobid>.log
#
# =============================================================================

# Set environment variables to prevent BLAS oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Unlimited CPU time for processes
ulimit -t unlimited

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "Start Time:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Change to the directory from which the job was submitted
cd "$SLURM_SUBMIT_DIR" || {
    echo "ERROR: Failed to change to submission directory: $SLURM_SUBMIT_DIR"
    exit 1
}

echo "Working directory: $(pwd)"
echo ""

# Check if required files exist
if [ ! -f "run_cluster_experiment.py" ]; then
    echo "ERROR: run_cluster_experiment.py not found"
    exit 1
fi

if [ ! -f "run_cluster_factor_experiment.py" ]; then
    echo "ERROR: run_cluster_factor_experiment.py not found"
    exit 1
fi

echo "All required files found."
echo ""

# Print Python information
echo "Python version:"
python --version
echo ""

# Print installed packages (optional, comment out if too verbose)
echo "Key installed packages:"
pip list | grep -E "numpy|pandas|networkx|matplotlib|joblib|openpyxl|scikit-learn" || echo "Package check skipped"
echo ""

# Run the experiment
echo "=========================================="
echo "Starting Cluster Factor Experiment"
echo "=========================================="
echo ""

python -u run_cluster_experiment.py || {
    echo ""
    echo "=========================================="
    echo "ERROR: Experiment failed"
    echo "=========================================="
    echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
    exit 1
}

echo ""
echo "=========================================="
echo "Experiment Completed Successfully"
echo "=========================================="
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# List output files
echo "Output files generated:"
ls -lh cluster_results/ 2>/dev/null | head -20 || echo "No results directory yet"
echo ""
ls -lh experiment_output_*.log 2>/dev/null || echo "No output log files"
echo ""

# Optional: Create a completion marker file
touch .job_${SLURM_JOB_ID}_complete

exit 0

