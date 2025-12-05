import os
import random
import math
import numpy as np
import pickle
import glob
from functools import partial
# Enforce single-threaded BLAS so no hidden parallel sums
os.environ.update({
  "OMP_NUM_THREADS":      "1",
  "MKL_NUM_THREADS":      "1",
  "OPENBLAS_NUM_THREADS": "1",
})

# Master seed for reproducibility
MASTER_SEED = 1996
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import copy
from joblib import Parallel, delayed
from typing import Any
from datetime import datetime
import time

timestamp = datetime.now().strftime("%m%d_%H%M")
#imports and reloads
from importlib import reload
import sys
sys.path.insert(0, '../src')
import SocialNetworkGenerator
import AgentManipulator
import NodeSelector
import Simulator
import Dicts_Lists_Helpers

reload(SocialNetworkGenerator)
reload(AgentManipulator)
reload(NodeSelector)
reload(Simulator)
reload(Dicts_Lists_Helpers)

import SocialNetworkGenerator as SNG
BaselineNetworkAnalyzer = SNG.BaselineNetworkAnalyzer
SocialNetworkAnalyzer   = SNG.SocialNetworkAnalyzer
from AgentManipulator import *
from NodeSelector import *
from Simulator import *
from Dicts_Lists_Helpers import *



def func1(val, val2):
    #return 3 * np.log(val + 10) + 2 + 0.0 * np.log(0.0001 * val2 + 1)
    return 3*(2*np.log(0.1*val+10) + 1*np.log(0.001*val2+1) -5) + 5


def custom_g(x1: float, x2: float, scaling_param: float = 30.0) -> float:

    re1 = 1 - (1 + np.exp(scaling_param * (x2 - x1) / x1)) ** (-1 / scaling_param)
    return re1


def custom_g1(x1, x2, scaling_param = 0.1):
    delta = x2 - x1
    alpha = 0.05 # controls the asymmetry — this replaces the 0.5 in your np.where
    smooth_factor = 1  # larger = sharper transition between the two regimes

    # Use a smooth transition between the two slopes
    weight = 1 / (1 + np.exp(-smooth_factor * delta))  # smooth approximation to 1(x2 > x1)

    slope = weight * scaling_param * alpha + (1 - weight) * scaling_param

    rew =  1 / (1 + np.exp(-slope * delta))
    return rew

def cosine_similarity_on_adults(G: nx.Graph, i: Any, j: Any) -> float:
    features = ['adults']
    diff = sum((G.nodes[i][feat] - G.nodes[j][feat])**2 for feat in features)
    return 1.0 / (0.1 + diff)  # example


def my_redmeat_sim(G: nx.Graph, i: Any, j: Any) -> float:
    r1 = G.nodes[i].get('redmeat', 0.0)
    r2 = G.nodes[j].get('redmeat', 0.0)
    return 1.0 / (1 + (r1 - r2)**2)

PICKLE_PATH = "agent.pickle" 


#################################################################################################
#################################################################################################

def run_mc_block_with_cluster_factor(
    label: str,
    df: pd.DataFrame,
    N: int,
    *,
    use_baseline: bool,
    capped: bool = False,
    n_runs: int = 5,
    avg_degree: float = 15.0,
    p_triangle: float = 0.35,
    reorient_edges_p: float = 0.0,
    visualize: bool = True,
    # simulation param grids
    alpha_0_list = (0.5,),
    alpha_list   = (0.3,),
    beta_list    = (5.0,),
    sit_list     = (0.005,),
    x_list       = None,             # if None, build default power_partition(...)
    use_random_bools = (False,),     # drives NodeSelector randomness
    q_list = (1,),                   # cluster indices
    cluster_factor_list = (0.3, 0.5, 0.7, 1.0),  # NEW: cluster similarity factor
    lancet_pct_list = (0.0, 0.05, 0.10, 0.15, 0.20),  # NEW: lancet adoption percentages
    n_jobs: int = 8,
    root_dir: str = None,
    timestamp: str = None,
):
    """
    Run one Monte Carlo experiment with cluster factor and lancet adoption percentage variations.

    Parameters
    ----------
    label : str
        Name used in output paths & plots.
    df : DataFrame
        Shared household sample (identical across experiments for comparability).
    N : int
        Number of households (=df.shape[0]) used to scale emissions.
    use_baseline : bool
        If True, use BaselineNetworkAnalyzer; else SocialNetworkAnalyzer (data-driven).
    capped : bool
        Passed to BaselineNetworkAnalyzer.create_and_analyze_graph(capped=...).
    n_runs : int
        Monte Carlo replicates (independent network draws).
    avg_degree, p_triangle, reorient_edges_p, visualize
        Passed into network creation for baseline case.
    *_list
        Simulation parameter grids.
    x_list : iterable or None
        Static-fraction parameter values. If None, defaults via power_partition(0,1,7,p=3).
    use_random_bools, q_list
        Drive NodeSelector choices.
    cluster_factor_list : tuple
        Different cluster similarity factors to test (when agents from different clusters).
    lancet_pct_list : tuple
        Different percentages of population to impose lancet diet (0.0 = 0%, 1.0 = 100%).
    n_jobs : int
        Parallel jobs in joblib.
    root_dir : str or None
        Root output directory. If None, "MC_runs_<timestamp>" created in cwd.
    timestamp : str or None
        Used in filenames. If None, generated now.

    Returns
    -------
    df_all : DataFrame
        All run/param rows concatenated (includes run index).
    df_std : DataFrame
        Summary (mean, std, low, high) by x_bin.
    out_dir : str
        Experiment output folder.
    """

    # ------------------------------------------------------------------
    # timestamp & output folders
    # ------------------------------------------------------------------
    if timestamp is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
    if root_dir is None:
        root_dir = f"MC_runs_{timestamp}"
    os.makedirs(root_dir, exist_ok=True)

    out_dir = os.path.join(root_dir, label)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # static x grid
    # ------------------------------------------------------------------
    if x_list is None:
        # you already have power_partition imported
        x_list = power_partition(0.0, 1.0, 7, p=3)

    # ------------------------------------------------------------------
    # container for per-run results
    # ------------------------------------------------------------------
    Cluster_MC = {}

    # ------------------------------------------------------------------
    # Monte Carlo loop
    # ------------------------------------------------------------------
    for i in range(n_runs):
        seed_i = MASTER_SEED + i         # just a label; we do NOT reseed global RNG inside loop

        # --- build network for this run ---
        if use_baseline:
            analyzer = BaselineNetworkAnalyzer(df, seed=seed_i)
            prefix = f"{label}_run{i}_{timestamp}"
            analyzer.plot_prefix = prefix
            G = analyzer.create_and_analyze_graph(
                avg_degree      = avg_degree,
                p_triangle      = p_triangle,
                reorient_edges_p= reorient_edges_p,
                visualize       = visualize,
                capped          = capped,
                tseed           = seed_i,
            )
        else:
            # data-driven SocialNetworkAnalyzer path
            analyzer = SocialNetworkAnalyzer(df, "cult1", "income")
            prefix = f"{label}_run{i}_{timestamp}"
            analyzer.plot_prefix = prefix
            G = analyzer.create_and_analyze_graph(
                func1,                      # your HH degree function
                visualize=visualize
            )

        print(f"[{label}] run {i}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
       

        # build an agent population on this graph
        all_agents_obj = AllAgents(
            conversion_factors,
            emission_factors,
            G,
            basic_protein_cols
        )

        # ------------------------------------------------------------------
        # parameter sweep on this *fixed* network
        # ------------------------------------------------------------------
        results = pd.DataFrame()
        all_results_df = pd.DataFrame()
        
        # NEW: Loop over cluster factors
        for cluster_factor in cluster_factor_list:
            # NEW: Loop over lancet adoption percentages
            for lancet_pct in lancet_pct_list:
                
                for use_random_bool in use_random_bools:
                    for q in q_list:

                        t0 = time.perf_counter()
                        print(f"[{label}] run {i} cluster {q} random={use_random_bool} cluster_factor={cluster_factor} lancet_pct={lancet_pct:.1%}")


                        # 6) Build parameter combinations
                        param_combinations = []
                        for alpha_0 in alpha_0_list:
                            for alpha in alpha_list:
                                for beta in beta_list:
                                    for sit in sit_list:
                                        for x in x_list:
                                            param_combinations.append((alpha_0, alpha, beta, sit, x))

                        print("Loop about to start")
                        print(f"Total parameter combinations: {len(param_combinations)}")

                        def sim_func(params):
                            alpha_0, alpha, beta, sit, x = params
                            # Construct a fresh SimulationConfif
                            fresh_agents = copy.deepcopy(all_agents_obj)

                            # Create cluster similarity function with the current cluster_factor
                            cluster_sim_func = partial(
                                AllAgents.quantile_similarity,
                                same_cluster_factor=1.0,
                                diff_cluster_factor=cluster_factor
                            )

                            conf = SimulationConfif(
                                alpha_0, alpha, beta, sit,
                                g_func=custom_g,
                                hh_similarity_func=cosine_similarity_on_adults,
                                redmeat_similarity_func=my_redmeat_sim,
                                cluster_similarity_func=cluster_sim_func,
                                diet_imp=True,
                                prnt = False
                            )

                            select_nodes = NodeSelector()
                            G_fresh = fresh_agents.network
                            
                            # Select nodes for lancet diet based on percentage
                            if lancet_pct > 0:
                                # Get all nodes
                                all_nodes = list(G_fresh.nodes())
                                n_lancet = int(len(all_nodes) * lancet_pct)
                                
                                # Randomly select nodes for lancet diet
                                lancet_nodes = random.sample(all_nodes, n_lancet)
                                
                                # Also get static nodes based on cluster
                                staticnodes_dict = select_nodes.get_nodes_by_category(
                                    G_fresh, x, 'kmeans_cluster', use_random=use_random_bool
                                )
                                staticnodes = staticnodes_dict[q]
                                
                                # Combine: lancet nodes become static
                                combined_static = list(set(lancet_nodes) | set(staticnodes))
                            else:
                                # No lancet diet, just use cluster-based static nodes
                                staticnodes_dict = select_nodes.get_nodes_by_category(
                                    G_fresh, x, 'kmeans_cluster', use_random=use_random_bool
                                )
                                staticnodes = staticnodes_dict[q]
                                combined_static = staticnodes
                                lancet_nodes = []

                            fresh_agents.set_static_nodes(combined_static)
                            
                            # Impose lancet diet on selected nodes
                            if lancet_pct > 0:
                                impose_diet = ImposeDiet(
                                    conversion_factors,
                                    emission_factors,
                                    fresh_agents,
                                    basic_protein_cols,
                                    agentlist1=lancet_nodes
                                )
                                impose_diet.impose_lanclet_diet()
                        
                            run_sim_collect = RunSimulationCollectData()
                            results =  run_sim_collect.run_simulation(
                                objSimulationConfig=conf,
                                items=pro_share_list,       # or your actual list
                                objAllAgents=fresh_agents
                            )#optional static_nodes
                            fresh_agents.normalize_protein_shares(flag_error=True)
                            
                            # Add cluster_factor and lancet_pct to results
                            results['cluster_factor'] = cluster_factor
                            results['lancet_pct'] = lancet_pct
                            results['n_lancet_nodes'] = len(lancet_nodes)
                            
                            del fresh_agents, conf, run_sim_collect, select_nodes, staticnodes_dict
                            return results

                        # 7) Run the simulations

                        # 7) Run the simulations in parallel
                        results = Parallel(n_jobs=n_jobs)(
                            delayed(sim_func)(p) for p in param_combinations
                        )

                        # Filter out any None results, in case some parameter combos returned None
                        results = [res for res in results if res is not None]

                        results_df = pd.DataFrame(results)
                        #print("Simulations complete.")
                        #print(results_df.head())
                        results_df['cluster number'] = q
                        results_df['random'] = use_random_bool
 

                        
                        all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)

                        t1 = time.perf_counter()
                        print(f"run {i:02d} (cluster_factor={cluster_factor}, lancet_pct={lancet_pct:.1%}) took {(t1-t0)/60:.1f} min")

        all_results_df = all_results_df.fillna(0)
        all_results_df['net_final_denormalized_total_emission_from_food_total'] = all_results_df['dynamic_final_denormalized_total_emission_from_food_total']+all_results_df['static_final_denormalized_total_emission_from_food_total']

        # ** NEW **: write out this single-run table immediately
        # make sure the output folder exists


        
        single_out = os.path.join(out_dir, f"MC_cluster_run_{i:02d}_{timestamp}.xlsx")
        all_results_df.to_excel(single_out, index=False)
        print(f"saved run {i} results to {single_out}")
        
        Cluster_MC[i] = all_results_df

    # ------------------------------------------------------------------
    # aggregate across runs
    # ------------------------------------------------------------------
    df_all = (
        pd.concat(Cluster_MC, names=["run", None], ignore_index=False)
          .reset_index(level="run")
          .reset_index(drop=True)
    )

    # per-household scaling
    df_all['avg_emission'] = (
        df_all['net_final_denormalized_total_emission_from_food_total'] / N
    )

    # stable bin for x (avoid float fuzz)
    df_all['x_bin'] = df_all['x'].round(6)

    # Group by x_bin, cluster_factor, and lancet_pct
    df_std = (
        df_all.groupby(["x_bin", "cluster_factor", "lancet_pct"], as_index=False)['avg_emission']
              .agg(mean="mean", std="std")
    )
    df_std['low']  = df_std['mean'] - df_std['std']
    df_std['high'] = df_std['mean'] + df_std['std']

    # ------------------------------------------------------------------
    # plot summary for each cluster_factor
    # ------------------------------------------------------------------
    for cluster_factor in cluster_factor_list:
        df_cluster = df_std[df_std['cluster_factor'] == cluster_factor]
        
        fig, axes = plt.subplots(1, len(lancet_pct_list), figsize=(20, 4), sharey=True)
        if len(lancet_pct_list) == 1:
            axes = [axes]
        
        for idx, lancet_pct in enumerate(lancet_pct_list):
            df_plot = df_cluster[df_cluster['lancet_pct'] == lancet_pct]
            ax = axes[idx]
            ax.fill_between(df_plot["x_bin"], df_plot["low"], df_plot["high"],
                           color="steelblue", alpha=0.25, label="Mean ± 1 SD")
            ax.plot(df_plot["x_bin"], df_plot["mean"], "o-", color="steelblue", lw=2, label="Mean")
            ax.set_xlabel("x (fraction static)")
            ax.set_title(f"Lancet adoption: {lancet_pct:.0%}")
            ax.grid(True, linestyle="--", alpha=0.6)
            if idx == 0:
                ax.set_ylabel("Avg. household emission")
                ax.legend()
        
        fig.suptitle(f"{label} – Cluster factor: {cluster_factor}", fontsize=14, y=1.02)
        plt.tight_layout()
        fig_path = os.path.join(out_dir, f"{label}_cluster_factor_{cluster_factor}_{timestamp}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[{label}] summary plot for cluster_factor={cluster_factor} → {fig_path}")

    # summary table
    tab_path = os.path.join(out_dir, f"{label}_avg_emission_by_params_{timestamp}.xlsx")
    df_std.to_excel(tab_path, index=False)
    print(f"[{label}] summary table → {tab_path}")

    return df_all, df_std, out_dir

#################################################################################################
#################################################################################################




#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################

#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################

if __name__ == "__main__":
    # common data sample (do this once upstream)
    df_dummy_org = pd.read_csv("../data/preprocessedconsumptions4_ICM.csv")
    fraction = 0.01
    df_dummy = df_dummy_org.sample(frac=fraction, random_state=MASTER_SEED)
    N = df_dummy.shape[0]

    timestamp = datetime.now().strftime("%m%d_%H%M")
    root_dir  = f"MC_runs_{timestamp}"

    # Run with cluster factor variations
    # Test with baseline_capped first (you can add more later)
    df_all_cluster, df_std_cluster, dir_cluster = run_mc_block_with_cluster_factor(
        label="cluster_factor_experiment",
        df=df_dummy,
        N=N,
        use_baseline=True,
        capped=True,
        n_runs=2,  # Start with 2 runs for testing
        cluster_factor_list=(0.3, 0.5, 0.7, 1.0),  # Different cluster similarity factors
        lancet_pct_list=(0.0, 0.05, 0.10, 0.15, 0.20),  # 0%, 5%, 10%, 15%, 20%
        root_dir=root_dir,
        timestamp=timestamp,
        n_jobs=8,
    )

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"Results saved in: {dir_cluster}")
    print(f"Summary statistics saved in: {os.path.join(dir_cluster, f'cluster_factor_experiment_avg_emission_by_params_{timestamp}.xlsx')}")

