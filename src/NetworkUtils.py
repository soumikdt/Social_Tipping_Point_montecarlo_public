"""
NetworkUtils.py

Utility functions for loading networks from pickle and reducing node counts
for local testing.
"""

import pickle
import networkx as nx
import random
import numpy as np
from typing import Optional, Tuple
from AgentManipulator import AllAgents


def load_agents_from_pickle(
    pickle_path: str = "agent.pickle",
    reduce_nodes: Optional[float] = None,
    seed: int = 1996
) -> AllAgents:
    """
    Load AllAgents object from pickle file.
    
    Parameters
    ----------
    pickle_path : str
        Path to the pickle file
    reduce_nodes : Optional[float]
        If provided, reduce network to this fraction of nodes (e.g., 0.01 = 1%)
        Useful for local testing before sending to cluster
    seed : int
        Random seed for node selection when reducing
        
    Returns
    -------
    AllAgents
        Loaded AllAgents object (possibly with reduced network)
    """
    print(f"Loading agents from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        all_agents_obj = pickle.load(f)
    
    original_n = all_agents_obj.network.number_of_nodes()
    original_e = all_agents_obj.network.number_of_edges()
    
    if reduce_nodes is not None and reduce_nodes < 1.0:
        print(f"Reducing network from {original_n} nodes to {int(original_n * reduce_nodes)} nodes ({reduce_nodes:.1%})...")
        all_agents_obj = reduce_network_size(all_agents_obj, fraction=reduce_nodes, seed=seed)
        new_n = all_agents_obj.network.number_of_nodes()
        new_e = all_agents_obj.network.number_of_edges()
        print(f"Reduced network: {new_n} nodes, {new_e} edges")
    else:
        print(f"Network loaded: {original_n} nodes, {original_e} edges")
        
    return all_agents_obj


def reduce_network_size(
    all_agents_obj: AllAgents,
    fraction: float = 0.01,
    seed: int = 1996
) -> AllAgents:
    """
    Reduce network size by keeping a random sample of nodes and their edges.
    
    Parameters
    ----------
    all_agents_obj : AllAgents
        Original AllAgents object
    fraction : float
        Fraction of nodes to keep (e.g., 0.01 = 1%)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    AllAgents
        New AllAgents object with reduced network
    """
    random.seed(seed)
    np.random.seed(seed)
    
    G = all_agents_obj.network
    all_nodes = list(G.nodes())
    n_keep = max(1, int(len(all_nodes) * fraction))
    
    # Randomly sample nodes to keep
    nodes_to_keep = set(random.sample(all_nodes, n_keep))
    
    # Create subgraph with only these nodes and edges between them
    G_reduced = G.subgraph(nodes_to_keep).copy()
    
    # Create new AllAgents object with reduced network
    from AgentManipulator import AllAgents
    from Dicts_Lists_Helpers import conversion_factors, emission_factors, basic_protein_cols
    
    # Preserve static nodes that are still in the reduced network
    static_nodes_reduced = [n for n in all_agents_obj._AM_static_nodes if n in nodes_to_keep]
    
    all_agents_reduced = AllAgents(
        conversion_factors,
        emission_factors,
        G_reduced,
        basic_protein_cols,
        stat_node=static_nodes_reduced
    )
    
    return all_agents_reduced


def get_network_stats(all_agents_obj: AllAgents) -> dict:
    """Get basic statistics about the network"""
    G = all_agents_obj.network
    return {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'n_static': len(all_agents_obj._AM_static_nodes),
        'n_dynamic': G.number_of_nodes() - len(all_agents_obj._AM_static_nodes),
        'avg_degree': 2 * G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    }

