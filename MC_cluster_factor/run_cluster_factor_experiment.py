"""
run_cluster_factor_experiment.py

Notebook-friendly script for cluster factor experiments.
- Loads from existing agent.pickle file
- Supports node reduction for local testing
- Uses streaming logger (writes directly to disk)
- Produces same output format as existing experiments
- Can be called from Jupyter notebook
"""

import os
import sys
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import copy
from functools import partial
from datetime import datetime
import time
from typing import Any, Dict, List, Optional, Callable
from joblib import Parallel, delayed

# Add src to path
sys.path.insert(0, '../src')

# Enforce single-threaded BLAS
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
})

# Master seed
MASTER_SEED = 1996 
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)

# Imports
from importlib import reload
import SocialNetworkGenerator
import AgentManipulator
import NodeSelector
import Simulator
import Dicts_Lists_Helpers
import sys
sys.path.insert(0, '../src')
from StreamingLogger import StreamingLogger
from NetworkUtils import load_agents_from_pickle, get_network_stats

reload(SocialNetworkGenerator)
reload(AgentManipulator)
reload(NodeSelector)
reload(Simulator)
reload(Dicts_Lists_Helpers)

from AgentManipulator import *
from NodeSelector import *
from Simulator import *
from Dicts_Lists_Helpers import *


def custom_g(x1: float, x2: float, scaling_param: float = 30.0) -> float:
    re1 = 1 - (1 + np.exp(scaling_param * (x2 - x1) / x1)) ** (-1 / scaling_param)
    return re1


def cosine_similarity_on_adults(G: nx.Graph, i: Any, j: Any) -> float:
    features = ['adults']
    diff = sum((G.nodes[i][feat] - G.nodes[j][feat])**2 for feat in features)
    return 1.0 / (0.1 + diff)


def my_redmeat_sim(G: nx.Graph, i: Any, j: Any) -> float:
    r1 = G.nodes[i].get('redmeat', 0.0)
    r2 = G.nodes[j].get('redmeat', 0.0)
    return 1.0 / (1 + (np.abs(r1 - r2))**(0.5))


def default_degree_function(val: float, val2: float) -> float:
    """
    Default degree function for SocialNetworkAnalyzer when constructing graphs
    from the preprocessed consumption CSV.
    Mirrors the heuristic used in earlier Monte Carlo scripts.
    """
    return 3 * (2 * np.log(0.1 * val + 10.0) + np.log(0.001 * val2 + 1.0) - 5.0) + 5.0



def run_cluster_factor_experiment(
    pickle_path: str = "/lu/topola/home/soumik96/simulation/montecarlosim/output_and_log/agent.pickle",
    reduce_nodes: Optional[float] = None,  # e.g., 0.01 for 1% (local testing)
    cluster_factor_list: tuple = (0.3, 0.5, 0.7, 1.0),
    alpha_0_list: tuple = (0.5,),
    alpha_list: tuple = (0.3,),
    beta_list: tuple = (5.0,),
    sit_list: tuple = (0.005,),
    x_list: Optional[list] = None,
    use_random_bools: tuple = (True,),
    q_list: tuple = (1,2,3),
    n_runs: int = 2,
    enable_logging: bool = True,
    log_dir: str = "simulation_logs",
    output_dir: str = None,
    timestamp: str = None,
    return_results: bool = True,
    n_jobs: int = 8,
    csv_path: Optional[str] = None,
    sample_fraction: Optional[float] = None,
    analyzer_label: Optional[str] = None,
    analyzer_visualize: bool = False,
    analyzer_primary_col: str = "cult1",
    analyzer_secondary_col: str = "income",
    degree_function: Optional[Callable[[float, float], float]] = None
):
    """
    Run cluster factor experiment with comprehensive logging.
    
    Parameters
    ----------
    pickle_path : str
        Path to agent.pickle file
    reduce_nodes : Optional[float]
        If provided, reduce network to this fraction (e.g., 0.01 = 1%)
        Use for local testing before cluster run
    cluster_factor_list : tuple
        Cluster similarity factors to test
    alpha_0_list, alpha_list, beta_list, sit_list : tuple
        Simulation parameters
    x_list : Optional[list]
        Static fraction values (also used as Lancet adoption percentages).
        If None, uses power_partition(0,1,7,p=3)
    use_random_bools, q_list : tuple
        Node selection parameters
    n_runs : int
        Number of Monte Carlo runs
    enable_logging : bool
        Enable comprehensive logging (writes to disk)
    log_dir : str
        Directory for log files
    output_dir : str
        Output directory for results
    timestamp : str
        Timestamp for output files
    return_results : bool
        Whether to return results (for notebook use)
    n_jobs : int
        Number of parallel workers for parameter combinations (joblib). Use 1 to run sequentially.
    csv_path : Optional[str]
        If provided, load the network from a preprocessed consumption CSV instead of a pickle.
    sample_fraction : Optional[float]
        Fraction of rows to sample from the CSV before building the network (e.g., 0.1 = 10%).
    analyzer_label : Optional[str]
        Label prefix for analyzer plots/logs when building from CSV.
    analyzer_visualize : bool
        Whether to enable visualisations inside SocialNetworkAnalyzer.
    analyzer_primary_col : str
        Primary column used by SocialNetworkAnalyzer for degree computation (default 'cult1').
    analyzer_secondary_col : str
        Secondary column (default 'income') passed to the degree function.
    degree_function : Optional[Callable[[float, float], float]]
        Custom degree function for SocialNetworkAnalyzer. Defaults to `default_degree_function`.
        
    Returns
    -------
    dict with keys:
        - df_all: All results DataFrame
        - df_std: Summary statistics DataFrame
        - out_dir: Output directory path
        - network_stats: Network statistics
    """
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = f"MC_runs_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp to log directory to prevent overwriting
    log_dir = f"{log_dir}_{timestamp}"
    
    if csv_path:
        print("="*80)
        print("LOADING AGENTS FROM CSV")
        print("="*80)
        print(f"Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from CSV")
        if sample_fraction is not None and 0 < sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=MASTER_SEED+ + datetime.now().second, replace= True).reset_index(drop=True)
            print(f"Sampled down to {len(df)} rows ({sample_fraction:.1%})")
        else:
            df = df.reset_index(drop=True)
        if reduce_nodes is not None and reduce_nodes < 1.0:
            print("NOTE: 'reduce_nodes' is ignored when building from CSV (use 'sample_fraction' instead).")
        degree_func = degree_function or default_degree_function
        analyzer = SocialNetworkAnalyzer(df, analyzer_primary_col, analyzer_secondary_col, seed=MASTER_SEED + + datetime.now().second)
        label = analyzer_label or Path(csv_path).stem
        analyzer.plot_prefix = f"{label}_{timestamp}"
        G_built = analyzer.create_and_analyze_graph(degree_func, visualize=analyzer_visualize)
        print(f"Constructed network from CSV: {G_built.number_of_nodes()} nodes, {G_built.number_of_edges()} edges")
        all_agents_obj = AllAgents(conversion_factors, emission_factors, G_built, basic_protein_cols)
        all_agents_obj.normalize_protein_shares()
        all_agents_obj.recalc_node_attr_from_pro_share()
        del df
    else:
        print("="*80)
        print("LOADING AGENTS FROM PICKLE")
        print("="*80)
        
        all_agents_obj = load_agents_from_pickle(
            pickle_path=pickle_path,
            reduce_nodes=reduce_nodes,
            seed=MASTER_SEED
        )
    
    network_stats = get_network_stats(all_agents_obj)
    print(f"\nNetwork statistics:")
    for k, v in network_stats.items():
        print(f"  {k}: {v}")
    
    N = network_stats['n_nodes']
    
    # Setup x_list if not provided
    if x_list is None:
        x_list = power_partition(0.0, 1.0, 7, p=3)
    
    # Container for results
    Cluster_MC = {}
    
    print("\n" + "="*80)
    print("STARTING MONTE CARLO RUNS")
    print("="*80)
    
    # Monte Carlo loop
    for run_idx in range(n_runs):
        print(f"\n{'='*80}")
        print(f"RUN {run_idx + 1}/{n_runs}")
        print(f"{'='*80}")
        
        # Create fresh copy for this run
        fresh_agents = copy.deepcopy(all_agents_obj)
        G = fresh_agents.network
        
        all_results_df = pd.DataFrame()
        
        # Loop over parameters
        for cluster_factor in cluster_factor_list:
            for use_random_bool in use_random_bools:
                for q in q_list:
                    
                    print(f"\n  cluster_factor={cluster_factor}, "
                          f"random={use_random_bool}, cluster={q}")
                    
                    # Prepare logging base directory
                    if enable_logging:
                        run_log_dir = os.path.join(
                            log_dir,
                            f"run{run_idx}_cf{cluster_factor}_rand{use_random_bool}_q{q}"
                        )
                        os.makedirs(run_log_dir, exist_ok=True)
                    else:
                        run_log_dir = None

                    # Build parameter combinations
                    param_combinations = []
                    for alpha_0 in alpha_0_list:
                        for alpha in alpha_list:
                            for beta in beta_list:
                                for sit in sit_list:
                                    for x in x_list:
                                        param_combinations.append((alpha_0, alpha, beta, sit, x))
                    
                    print(f"    Running {len(param_combinations)} parameter combinations...")
                    t0 = time.time()
                    
                    def run_single_param(combo_idx: int, params):
                        alpha_0, alpha, beta, sit, x = params
                        lancet_pct = x

                        # Deterministic seeding per combination
                        seed_offset = hash((run_idx, cluster_factor, use_random_bool, q, combo_idx, params)) % (2**32 - 1)
                        combo_seed = (MASTER_SEED + seed_offset+ + datetime.now().second) % (2**32 - 1)
                        random.seed(combo_seed)
                        np.random.seed(combo_seed)

                        sim_agents = copy.deepcopy(fresh_agents)

                        cluster_sim_func = partial(
                            AllAgents.quantile_similarity,
                            same_cluster_factor=1.0,
                            diff_cluster_factor=cluster_factor
                        )

                        logger_instance = None
                        if enable_logging and run_log_dir is not None:
                            combo_log_dir = os.path.join(run_log_dir, f"combo_{combo_idx:04d}")
                            os.makedirs(combo_log_dir, exist_ok=True)
                            logger_instance = StreamingLogger(
                                output_dir=combo_log_dir,
                                log_weights=True,
                                log_node_states=False,
                                log_iterations=True,
                                log_influence=False
                            )
                            logger_instance.set_metadata(
                                run=run_idx,
                                cluster_factor=cluster_factor,
                                use_random=use_random_bool,
                                cluster=q,
                                n_nodes=N,
                                n_edges=fresh_agents.network.number_of_edges(),
                                alpha_0=alpha_0,
                                alpha=alpha,
                                beta=beta,
                                sit=sit,
                                lancet_pct=lancet_pct
                            )

                        conf = SimulationConfif(
                            alpha_0, alpha, beta, sit,
                            g_func=custom_g,
                            hh_similarity_func=cosine_similarity_on_adults,
                            redmeat_similarity_func=my_redmeat_sim,
                            cluster_similarity_func=cluster_sim_func,
                            diet_imp=True,
                            prnt=False,
                            logger=logger_instance
                        )

                        G_sim = sim_agents.network
                        all_nodes = list(G_sim.nodes())

                        if lancet_pct > 0:
                            n_lancet = int(len(all_nodes) * lancet_pct)
                            lancet_nodes = random.sample(all_nodes, n_lancet)
                        else:
                            lancet_nodes = []

                        select_nodes = NodeSelector()
                        staticnodes_raw = select_nodes.get_nodes_by_any_quantiles(
                            G_sim, x, variable=[ 'Beef and veal aggregated protein share','Pork aggregated protein share','Lamb and goat aggregated protein share','Poultry aggregated protein share'], num_quantiles=3, use_random=use_random_bool
                        )
                        #print(staticnodes_raw)
                        if isinstance(staticnodes_raw, dict):
                            staticnodes = staticnodes_raw[f"Q{q}"]
                        else:
                            staticnodes = staticnodes_raw

                        combined_static = list(set(lancet_nodes) | set(staticnodes))
                        sim_agents.set_static_nodes(combined_static)

                        if lancet_pct > 0:
                            impose_diet = ImposeDiet(
                                conversion_factors,
                                emission_factors,
                                sim_agents,
                                basic_protein_cols,
                                agentlist1=lancet_nodes
                            )
                            impose_diet.impose_lanclet_diet()

                        run_sim_collect = RunSimulationCollectData()
                        results = run_sim_collect.run_simulation(
                            objSimulationConfig=conf,
                            items=pro_share_list,
                            objAllAgents=sim_agents
                        )

                        results['cluster_factor'] = cluster_factor
                        results['lancet_pct'] = lancet_pct
                        results['n_lancet_nodes'] = len(lancet_nodes)
                        results['run'] = run_idx
                        results['combo_index'] = combo_idx

                        if logger_instance:
                            logger_instance.close()

                        del sim_agents, conf, run_sim_collect
                        return results

                    if n_jobs and n_jobs > 1:
                        print("parallel"*10)
                        results_list = Parallel(n_jobs=n_jobs)(
                            delayed(run_single_param)(idx, params)
                            for idx, params in enumerate(param_combinations)
                        )
                    else:
                        print("nonparallel"*10)
                        results_list = [
                            run_single_param(idx, params)
                            for idx, params in enumerate(param_combinations)
                        ]
                    
                    if enable_logging and run_log_dir is not None:
                        print(f"    Logs saved under: {run_log_dir}")
                    
                    # Create DataFrame
                    results_df = pd.DataFrame(results_list)
                    results_df['cluster number'] = q
                    results_df['random'] = use_random_bool
                    
                    all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)
                    
                    t1 = time.time()
                    print(f"    Completed in {(t1-t0)/60:.1f} minutes")
        
        # Process results
        all_results_df = all_results_df.fillna(0)
        all_results_df['net_final_denormalized_total_emission_from_food_total'] = (
            all_results_df['dynamic_final_denormalized_total_emission_from_food_total'] +
            all_results_df['static_final_denormalized_total_emission_from_food_total']
        )
        
        # Save run results
        run_out_dir = os.path.join(output_dir, f"run_{run_idx:02d}")
        os.makedirs(run_out_dir, exist_ok=True)
        run_file = os.path.join(run_out_dir, f"MC_cluster_run_{run_idx:02d}_{timestamp}.xlsx")
        all_results_df.to_excel(run_file, index=False)
        print(f"\n  Saved run {run_idx} results to {run_file}")
        
        Cluster_MC[run_idx] = all_results_df
    
    # Aggregate across runs
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)
    
    # Concatenate with run index
    df_all = pd.concat(Cluster_MC, names=["run_index", None], ignore_index=False)
    
    # Reset index - handle case where 'run' column might already exist
    if 'run' in df_all.columns:
        # If 'run' column exists, use it and drop the index level
        df_all = df_all.reset_index(level="run_index", drop=True)
    else:
        # If 'run' column doesn't exist, create it from index
        df_all = df_all.reset_index(level="run_index")
        df_all = df_all.rename(columns={'run_index': 'run'})
    
    df_all = df_all.reset_index(drop=True)
    
    # Per-household scaling
    df_all['avg_emission'] = (
        df_all['net_final_denormalized_total_emission_from_food_total'] / N
    )
    
    # Stable bin for x
    df_all['x_bin'] = df_all['x'].round(6)
    
    # Summary statistics
    df_std = (
        df_all.groupby(["x_bin", "cluster_factor", "lancet_pct"], as_index=False)['avg_emission']
              .agg(mean="mean", std="std")
    )
    
    # Handle NaN std (happens when n_runs=1, since std of single value is undefined)
    # Fill NaN std with 0 so plots work correctly
    n_nan_std = df_std['std'].isna().sum()
    if n_nan_std > 0:
        print(f"\nNote: {n_nan_std} groups have NaN std (likely n_runs=1). Setting std=0 for these groups.")
    df_std['std'] = df_std['std'].fillna(0)
    
    df_std['low'] = df_std['mean'] - df_std['std']
    df_std['high'] = df_std['mean'] + df_std['std']
    
    # Save summary
    summary_file = os.path.join(output_dir, f"cluster_factor_experiment_summary_{timestamp}.xlsx")
    df_std.to_excel(summary_file, index=False)
    print(f"Saved summary to {summary_file}")
    
    # Create plots
    print("\nCreating plots...")
    for cluster_factor in cluster_factor_list:
        df_cluster = df_std[df_std['cluster_factor'] == cluster_factor]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot with lancet_pct on x-axis (which equals x_bin)
        ax.fill_between(df_cluster["lancet_pct"], df_cluster["low"], df_cluster["high"],
                       color="steelblue", alpha=0.25, label="Mean ± 1 SD")
        ax.plot(df_cluster["lancet_pct"], df_cluster["mean"], "o-", color="steelblue", lw=2, label="Mean")
        ax.set_xlabel("Lancet adoption (fraction of nodes)")
        ax.set_ylabel("Avg. household emission")
        ax.set_title(f"Cluster factor: {cluster_factor}", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"cluster_factor_{cluster_factor}_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {plot_file}")
    
    # Combined plot across cluster factors
    if not df_std.empty:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for cluster_factor in sorted(df_std['cluster_factor'].unique()):
            df_cluster = df_std[df_std['cluster_factor'] == cluster_factor].sort_values('lancet_pct')
            if df_cluster.empty:
                continue
            ax.fill_between(
                df_cluster["lancet_pct"],
                df_cluster["low"],
                df_cluster["high"],
                alpha=0.15,
                label=f"{cluster_factor} ± 1 SD"
            )
            ax.plot(
                df_cluster["lancet_pct"],
                df_cluster["mean"],
                marker="o",
                lw=2,
                label=f"{cluster_factor} mean"
            )
        ax.set_xlabel("Lancet adoption (fraction of nodes)")
        ax.set_ylabel("Avg. household emission")
        ax.set_title("Cluster Factor Comparison", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        combined_plot_file = os.path.join(output_dir, f"cluster_factor_comparison_{timestamp}.png")
        plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved combined plot: {combined_plot_file}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"Results directory: {output_dir}")
    print(f"Logs directory: {log_dir}")
    
    # Show log file locations
    if enable_logging and os.path.exists(log_dir):
        print(f"\nLog files are located in:")
        import glob
        log_subdirs = glob.glob(os.path.join(log_dir, "run*"))
        for subdir in sorted(log_subdirs):
            weight_files = glob.glob(os.path.join(subdir, "weight_logs_*.csv"))
            if weight_files:
                print(f"  {subdir}/")
                print(f"    - weight_logs_*.csv (weight calculations - CSV)")
                print(f"    - weight_logs_*.jsonl (weights - structured JSON via Python logging)")
                print(f"    - iteration_logs_*.csv / iteration_logs_*.jsonl (iteration statistics)")
                print(f"    - metadata_*.json / metadata_*.jsonl (run metadata)")
    
    if return_results:
        return {
            'df_all': df_all,
            'df_std': df_std,
            'out_dir': output_dir,
            'network_stats': network_stats
        }


if __name__ == "__main__":
    # Example usage
    results = run_cluster_factor_experiment(
        pickle_path="agent.pickle",
        reduce_nodes=0.01,  # 1% for local testing
        n_runs=1,
        enable_logging=True
    )

