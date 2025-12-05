#!/usr/bin/env python3
"""
run_cluster_experiment.py

Production script for running cluster factor experiments on HPC cluster.
This script loads the full agent.pickle (no network reduction) and runs
the complete Monte Carlo experiment with comprehensive logging.

Usage:
    python run_cluster_experiment.py

Configuration:
    Edit the parameters in the main() function below to customize the experiment.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
from datetime import datetime
import time
import logging

# Enforce single-threaded BLAS to avoid oversubscription
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
})

# Master seed for reproducibility
MASTER_SEED = 1996 + datetime.now().second
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

# Add src to path
sys.path.insert(0, '../src')

# Import the experiment function
from run_cluster_factor_experiment import run_cluster_factor_experiment


class TeeLogger:
    """Logger that writes to both stdout and a file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', buffering=1)  # Line buffered
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def main():
    """
    Main function to run the cluster factor experiment.
    
    Configure your experiment parameters here.
    """
    
    # Generate timestamp for this experiment run
    timestamp_log = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory first (will be used for console log)
    output_dir_name = f"MC_runs_{timestamp_log}"
    os.makedirs(output_dir_name, exist_ok=True)
    
    # Setup output logging to file (inside the output directory)
    output_log_file = os.path.join(output_dir_name, "console_output.log")
    
    # Redirect stdout to both console and file
    tee = TeeLogger(output_log_file)
    sys.stdout = tee
    
    print("="*80)
    print("CLUSTER FACTOR EXPERIMENT - HPC RUN")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Timestamp: {timestamp_log}")
    print(f"Master seed: {MASTER_SEED}")
    print(f"Output directory: {output_dir_name}")
    print(f"Console log: {output_log_file}")
    print()
    
    # ==========================================================================
    # CONFIGURATION - Edit these parameters for your experiment
    # ==========================================================================
    
    # Path to agent pickle file (FULL NETWORK - no reduction)
    # Auto-detect: use cluster path if it exists, otherwise use local path
    CLUSTER_PICKLE_PATH = "/lu/topola/home/soumik96/montecarlosim/output_and_log/agent.pickle"
    LOCAL_PICKLE_PATH = Path(__file__).resolve().parents[1] / "output_and_log" / "agent.pickle"
    
    if os.path.exists(CLUSTER_PICKLE_PATH):
        PICKLE_PATH = CLUSTER_PICKLE_PATH
        print("Using cluster pickle path")
    elif LOCAL_PICKLE_PATH.exists():
        PICKLE_PATH = str(LOCAL_PICKLE_PATH)
        print("Using local pickle path")
    else:
        print("ERROR: Cannot find agent.pickle file!")
        print(f"  Tried cluster: {CLUSTER_PICKLE_PATH}")
        print(f"  Tried local: {LOCAL_PICKLE_PATH}")
        print("  Please update PICKLE_PATH in the script.")
        sys.exit(1)
    
    # Cluster factors to test (different cluster vs same cluster weight multiplier)
    # Using same values as local run for consistency
    CLUSTER_FACTORS = (0.0,1.0)
    
    # Number of Monte Carlo runs (increase for better statistics)
    N_RUNS = 3
    
    # Simulation parameters (usually keep these fixed)
    ALPHA_0 = (0.5,)  # Base adoption parameter
    ALPHA = (0.3,)    # Spending similarity weight
    BETA = (0.2,)     # Red meat similarity weight
    SIT = (0.005,)    # Spending increase tolerance
    
    # Static fraction values (also used as Lancet adoption percentages)
    # None = use power_partition(0, 1, 7, p=3) which gives ~9-10 values
    # Custom values: (0.0, 0.002, 0.02, 0.06, 0.15, 0.29, 0.5, 0.8)
    X_LIST = (0.0, 0.002, 0.02, 0.06, 0.15, 0.29, 0.5, 0.8)
    
    # Node selection parameters
    USE_RANDOM = (True,)   # Use random node selection
    Q_LIST = (1,2,3)          # Cluster number

    # Parallelism
    N_JOBS = 8             # Number of parallel workers for joblib

    # CSV-driven network construction
    CSV_PATH_LOCAL = "/Users/mimuw2022/Documents/GitHub/Social_Tipping_Point/simulation/montecarlosim/data/preprocessedconsumptions4_ICM.csv"
    CSV_PATH_CLUSTER = "/lu/topola/home/soumik96/montecarlosim/data/preprocessedconsumptions4_ICM.csv"
    CSV_PATH = None
    if os.path.exists(CSV_PATH_CLUSTER):
        CSV_PATH = CSV_PATH_CLUSTER
    elif os.path.exists(CSV_PATH_LOCAL):
        CSV_PATH = CSV_PATH_LOCAL
    print("CSV: ", CSV_PATH)
    SAMPLE_FRACTION = 0.1   # Use 10% sample when building network from CSV
    if not os.path.isfile(CSV_PATH):
        print(f"WARNING: CSV file not found at {CSV_PATH}. Falling back to pickle.")
        CSV_PATH = None
    
    # Print configuration
    print("="*80)
    print("EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Pickle path: {PICKLE_PATH}")
    if CSV_PATH:
        print(f"CSV path: {CSV_PATH} (sample_fraction={SAMPLE_FRACTION})")
    else:
        print("CSV path: <not available>")
    print(f"Cluster factors: {CLUSTER_FACTORS}")
    print(f"Monte Carlo runs: {N_RUNS}")
    print(f"X values (Lancet %): {X_LIST}")
    print(f"Random node selection: {USE_RANDOM[0]}")
    print(f"Alpha_0: {ALPHA_0[0]}, Alpha: {ALPHA[0]}, Beta: {BETA[0]}")
    print(f"Parallel workers (joblib): {N_JOBS}")
    print("="*80)
    print()
    
    # Logging configuration
    ENABLE_LOGGING = False
    LOG_DIR = "cluster_logs"
    
    # ==========================================================================
    # RUN EXPERIMENT
    # ==========================================================================
    
    start_time = time.time()
    
    try:
        results = run_cluster_factor_experiment(
            pickle_path=PICKLE_PATH,
            reduce_nodes=None,  # IMPORTANT: No network reduction for cluster run
            cluster_factor_list=CLUSTER_FACTORS,
            alpha_0_list=ALPHA_0,
            alpha_list=ALPHA,
            beta_list=BETA,
            sit_list=SIT,
            x_list=X_LIST,
            use_random_bools=USE_RANDOM,
            q_list=Q_LIST,
            n_runs=N_RUNS,
            enable_logging=ENABLE_LOGGING,
            log_dir=LOG_DIR,
            output_dir=output_dir_name,  # Use the timestamped directory
            timestamp=timestamp_log,  # Pass the timestamp
            n_jobs=N_JOBS,
            csv_path=CSV_PATH,
            sample_fraction=SAMPLE_FRACTION,
            return_results=True
        )
        
        elapsed_time = time.time() - start_time
        
        print()
        print("="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total elapsed time: {elapsed_time/3600:.2f} hours")
        print(f"Results directory: {results['out_dir']}")
        print(f"Logs directory: {LOG_DIR}_{timestamp_log}")
        print(f"Console log: {output_log_file}")
        print()
        
        # Print summary statistics
        if 'df_std' in results:
            print("Summary Statistics:")
            print(f"  Total parameter combinations: {len(results['df_std'])}")
            print(f"  Cluster factors tested: {sorted(results['df_std']['cluster_factor'].unique())}")
            print(f"  Lancet adoption percentages: {sorted(results['df_std']['lancet_pct'].unique())}")
            print()
            
            # Print detailed statistics
            print("Detailed Results by Cluster Factor:")
            print("-" * 80)
            for cf in sorted(results['df_std']['cluster_factor'].unique()):
                df_cf = results['df_std'][results['df_std']['cluster_factor'] == cf]
                print(f"\nCluster Factor = {cf}:")
                print(f"  Mean emission range: [{df_cf['mean'].min():.4f}, {df_cf['mean'].max():.4f}]")
                print(f"  Average std: {df_cf['std'].mean():.4f}")
                print(f"  Number of data points: {len(df_cf)}")
            print()
        
        # Close the output logger
        tee.close()
        sys.stdout = tee.terminal
        
        print(f"\nAll output saved to: {output_log_file}")
        
        return 0  # Success
        
    except Exception as e:
        print()
        print("="*80)
        print("EXPERIMENT FAILED")
        print("="*80)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        
        # Close the output logger
        try:
            tee.close()
            sys.stdout = tee.terminal
            print(f"\nError log saved to: {output_log_file}")
        except:
            pass
        
        return 1  # Failure


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

