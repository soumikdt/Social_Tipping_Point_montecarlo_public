"""
MC_cluster_factor_with_logging.py

Same as MC_cluster_factor.py but with comprehensive logging enabled.

WARNING: Logging generates LARGE files. Only use for:
1. Small test runs (few nodes, few iterations)
2. Debugging specific scenarios
3. Understanding weight dynamics

For production runs, use MC_cluster_factor.py without logging.
"""

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
from SimulationLogger import SimulationLogger, LightweightLogger  # NEW

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
    return 3*(2*np.log(0.1*val+10) + 1*np.log(0.001*val2+1) -5) + 5


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
    return 1.0 / (1 + (r1 - r2)**2)


if __name__ == "__main__":
    # Load small sample for testing with logging
    print("="*80)
    print("CLUSTER FACTOR EXPERIMENT WITH DETAILED LOGGING")
    print("="*80)
    print("\nWARNING: This will generate LARGE log files!")
    print("Only use for small test runs to understand the dynamics.\n")
    
    df_dummy_org = pd.read_csv("../data/preprocessedconsumptions4_ICM.csv")
    
    # Use very small fraction for logging test
    fraction = 0.001  # 0.1% of data - very small!
    df_dummy = df_dummy_org.sample(frac=fraction, random_state=MASTER_SEED)
    N = df_dummy.shape[0]
    
    print(f"Sample size: {N} households")
    print(f"This will run ONE simulation with full logging enabled.")
    print(f"Expected log size: ~100MB - 1GB depending on network size\n")
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    root_dir = f"MC_runs_with_logging_{timestamp}"
    os.makedirs(root_dir, exist_ok=True)
    
    # Create network
    print("Creating network...")
    analyzer = BaselineNetworkAnalyzer(df_dummy, seed=MASTER_SEED)
    G = analyzer.create_and_analyze_graph(
        avg_degree=15.0,
        p_triangle=0.35,
        reorient_edges_p=0.0,
        visualize=True,
        capped=True,
        tseed=MASTER_SEED,
    )
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create AllAgents
    all_agents_obj = AllAgents(
        conversion_factors,
        emission_factors,
        G,
        basic_protein_cols
    )
    
    # LOGGING CONFIGURATION
    # Sample only a few nodes for detailed logging to keep file size manageable
    sample_size = min(10, G.number_of_nodes())  # Log only 10 nodes in detail
    sample_nodes = list(G.nodes())[:sample_size]
    
    print(f"\n{'='*80}")
    print("LOGGING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Full logging for {sample_size} sample nodes")
    print(f"Summary logging for all nodes")
    print(f"Log directory: {root_dir}/simulation_logs/")
    print(f"{'='*80}\n")
    
    # Create logger
    logger = SimulationLogger(
        output_dir=os.path.join(root_dir, "simulation_logs"),
        log_weights=True,           # Log all weight calculations
        log_node_states=True,        # Log node states before/after updates
        log_iterations=True,         # Log iteration-level statistics
        log_influence=True,          # Log neighbor influence
        sample_nodes=sample_nodes,   # Only detailed logs for sample nodes
        save_format="csv"            # CSV format for easy inspection
    )
    
    # Set metadata
    logger.set_metadata(
        experiment="cluster_factor_with_logging",
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        sample_size=N,
        fraction=fraction,
        cluster_factor=0.5,
        lancet_pct=0.10,
        sample_nodes=sample_nodes
    )
    
    # Run single simulation
    print("Running simulation with logging...")
    t0 = time.time()
    
    cluster_factor = 0.5
    lancet_pct = 0.10
    
    fresh_agents = copy.deepcopy(all_agents_obj)
    
    # Create cluster similarity function
    cluster_sim_func = partial(
        AllAgents.quantile_similarity,
        same_cluster_factor=1.0,
        diff_cluster_factor=cluster_factor
    )
    
    # Create configuration WITH LOGGER
    conf = SimulationConfif(
        alpha_0=0.5,
        alpha=0.3,
        beta=5.0,
        sit=0.005,
        g_func=custom_g,
        hh_similarity_func=cosine_similarity_on_adults,
        redmeat_similarity_func=my_redmeat_sim,
        cluster_similarity_func=cluster_sim_func,
        diet_imp=True,
        prnt=True,
        logger=logger  # PASS LOGGER HERE
    )
    
    # Select nodes for lancet diet
    G_fresh = fresh_agents.network
    all_nodes = list(G_fresh.nodes())
    n_lancet = int(len(all_nodes) * lancet_pct)
    lancet_nodes = random.sample(all_nodes, n_lancet)
    
    # Set static nodes
    select_nodes = NodeSelector()
    staticnodes_dict = select_nodes.get_nodes_by_category(
        G_fresh, 0.1, 'kmeans_cluster', use_random=False
    )
    staticnodes = staticnodes_dict[1]
    combined_static = list(set(lancet_nodes) | set(staticnodes))
    fresh_agents.set_static_nodes(combined_static)
    
    # Impose lancet diet
    if lancet_pct > 0:
        impose_diet = ImposeDiet(
            conversion_factors,
            emission_factors,
            fresh_agents,
            basic_protein_cols,
            agentlist1=lancet_nodes
        )
        impose_diet.impose_lanclet_diet()
    
    # Run simulation
    run_sim_collect = RunSimulationCollectData()
    results = run_sim_collect.run_simulation(
        objSimulationConfig=conf,
        items=pro_share_list,
        objAllAgents=fresh_agents
    )
    
    t1 = time.time()
    print(f"\nSimulation completed in {t1-t0:.2f} seconds")
    
    # Save logs
    print(f"\n{'='*80}")
    print("SAVING LOGS...")
    print(f"{'='*80}")
    logger.save_logs(prefix="test_run")
    
    # Print weight summary
    print(f"\n{'='*80}")
    print("WEIGHT SUMMARY (Same vs Different Clusters)")
    print(f"{'='*80}")
    weight_summary = logger.get_weight_summary()
    print(weight_summary)
    
    # Print iteration summary
    print(f"\n{'='*80}")
    print("ITERATION SUMMARY")
    print(f"{'='*80}")
    iteration_summary = logger.get_iteration_summary()
    print(iteration_summary[['iteration', 'dynamic_emission_mean', 'dynamic_emission_std', 
                              'static_emission_mean', 'static_emission_std']].to_string())
    
    print(f"\n{'='*80}")
    print("LOGS SAVED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Location: {root_dir}/simulation_logs/")
    print(f"\nFiles created:")
    log_dir = os.path.join(root_dir, "simulation_logs")
    for f in os.listdir(log_dir):
        fpath = os.path.join(log_dir, f)
        size_mb = os.path.getsize(fpath) / (1024*1024)
        print(f"  - {f} ({size_mb:.2f} MB)")
    
    print(f"\n{'='*80}")
    print("HOW TO ANALYZE LOGS")
    print(f"{'='*80}")
    print("1. Weight logs: Shows all weight calculations with components")
    print("   - Compare weights for same_cluster=True vs False")
    print("   - See how cluster_similarity affects final weights")
    print("")
    print("2. Node state logs: Shows node attributes before/after each update")
    print("   - Track how emissions change over iterations")
    print("   - See protein share evolution")
    print("")
    print("3. Iteration logs: Aggregate statistics per iteration")
    print("   - Plot emission_mean over iterations")
    print("   - Compare dynamic vs static node emissions")
    print("")
    print("4. Influence logs: Shows neighbor influence on each update")
    print("   - See which neighbors had most influence")
    print("   - Track emission changes per update")
    print("")
    print("All logs are in CSV format - open in Excel or load with pandas!")
    print(f"{'='*80}\n")

