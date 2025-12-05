"""
SimulationLogger.py

Comprehensive logging system for tracking simulation dynamics, weights, 
and intermediate states during the social network food consumption model.
"""

import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import networkx as nx


class SimulationLogger:
    """
    Logs detailed information during simulation runs including:
    - Weight calculations (spending, household, redmeat, cluster similarities)
    - Per-iteration node states
    - Influence propagation
    - Aggregate statistics over time
    """
    
    def __init__(
        self, 
        output_dir: str = "simulation_logs",
        log_weights: bool = True,
        log_node_states: bool = True,
        log_iterations: bool = True,
        log_influence: bool = True,
        sample_nodes: Optional[List] = None,
        save_format: str = "json"  # "json", "csv", or "pickle"
    ):
        """
        Initialize the simulation logger.
        
        Parameters
        ----------
        output_dir : str
            Directory to save log files
        log_weights : bool
            Whether to log weight calculations
        log_node_states : bool
            Whether to log node states at each iteration
        log_iterations : bool
            Whether to log iteration-level statistics
        log_influence : bool
            Whether to log influence from neighbors
        sample_nodes : Optional[List]
            If provided, only log detailed info for these nodes (to reduce log size)
        save_format : str
            Format to save logs: "json", "csv", or "pickle"
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_weights = log_weights
        self.log_node_states = log_node_states
        self.log_iterations = log_iterations
        self.log_influence = log_influence
        self.sample_nodes = sample_nodes
        self.save_format = save_format
        
        # Storage for logs
        self.weight_logs = []
        self.node_state_logs = []
        self.iteration_logs = []
        self.influence_logs = []
        self.metadata = {}
        
        # Tracking
        self.current_iteration = 0
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
    def set_metadata(self, **kwargs):
        """Store metadata about the simulation run"""
        self.metadata.update(kwargs)
        self.metadata['run_id'] = self.run_id
        self.metadata['timestamp'] = datetime.now().isoformat()
        
    def log_weight_calculation(
        self,
        iteration: int,
        node: Any,
        neighbor: Any,
        spending_sim: float,
        household_sim: float,
        redmeat_sim: float,
        cluster_sim: float,
        alpha: float,
        beta: float,
        final_weight: float,
        node_cluster: Any = None,
        neighbor_cluster: Any = None
    ):
        """
        Log a single weight calculation between node and neighbor.
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        node : Any
            Node being updated
        neighbor : Any
            Neighbor influencing the node
        spending_sim : float
            g(spending) similarity value
        household_sim : float
            Household similarity value
        redmeat_sim : float
            Red meat similarity value
        cluster_sim : float
            Cluster similarity factor
        alpha : float
            Alpha parameter
        beta : float
            Beta parameter
        final_weight : float
            Final computed weight
        node_cluster : Any
            Cluster of node (optional)
        neighbor_cluster : Any
            Cluster of neighbor (optional)
        """
        if not self.log_weights:
            return
            
        # Only log if node is in sample (or sample is None = log all)
        if self.sample_nodes is not None and node not in self.sample_nodes:
            return
            
        log_entry = {
            'iteration': iteration,
            'node': str(node),
            'neighbor': str(neighbor),
            'spending_similarity': spending_sim,
            'household_similarity': household_sim,
            'redmeat_similarity': redmeat_sim,
            'cluster_similarity': cluster_sim,
            'alpha': alpha,
            'beta': beta,
            'base_weight': alpha * spending_sim * household_sim + beta * redmeat_sim,
            'final_weight': final_weight,
            'node_cluster': str(node_cluster) if node_cluster is not None else None,
            'neighbor_cluster': str(neighbor_cluster) if neighbor_cluster is not None else None,
            'same_cluster': node_cluster == neighbor_cluster if node_cluster is not None else None
        }
        self.weight_logs.append(log_entry)
        
    def log_node_state(
        self,
        iteration: int,
        node: Any,
        state: Dict[str, float],
        stage: str = "after_update"
    ):
        """
        Log the state of a node at a particular point.
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        node : Any
            Node identifier
        state : Dict[str, float]
            Node attributes (emissions, spending, protein shares, etc.)
        stage : str
            When this state was captured (e.g., "before_update", "after_update")
        """
        if not self.log_node_states:
            return
            
        if self.sample_nodes is not None and node not in self.sample_nodes:
            return
            
        log_entry = {
            'iteration': iteration,
            'node': str(node),
            'stage': stage,
            **{k: v for k, v in state.items() if isinstance(v, (int, float, str, bool))}
        }
        self.node_state_logs.append(log_entry)
        
    def log_iteration_stats(
        self,
        iteration: int,
        G: nx.Graph,
        static_nodes: List = None
    ):
        """
        Log aggregate statistics for the entire network at this iteration.
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        G : nx.Graph
            The network graph
        static_nodes : List
            List of static nodes
        """
        if not self.log_iterations:
            return
            
        static_nodes = static_nodes or []
        
        # Separate dynamic and static nodes
        dynamic_nodes = [n for n in G.nodes() if n not in static_nodes]
        
        def get_stats(nodes):
            if not nodes:
                return {}
            emissions = [G.nodes[n].get('total emission from food', 0) for n in nodes]
            spending = [G.nodes[n].get('total spending on food', 0) for n in nodes]
            redmeat = [G.nodes[n].get('redmeat', 0) for n in nodes]
            
            return {
                'n_nodes': len(nodes),
                'emission_mean': np.mean(emissions),
                'emission_std': np.std(emissions),
                'emission_min': np.min(emissions),
                'emission_max': np.max(emissions),
                'spending_mean': np.mean(spending),
                'spending_std': np.std(spending),
                'redmeat_mean': np.mean(redmeat),
                'redmeat_std': np.std(redmeat),
            }
        
        log_entry = {
            'iteration': iteration,
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'n_static': len(static_nodes),
            'n_dynamic': len(dynamic_nodes),
        }
        
        # Add dynamic node stats
        dynamic_stats = get_stats(dynamic_nodes)
        for k, v in dynamic_stats.items():
            log_entry[f'dynamic_{k}'] = v
            
        # Add static node stats
        static_stats = get_stats(static_nodes)
        for k, v in static_stats.items():
            log_entry[f'static_{k}'] = v
            
        self.iteration_logs.append(log_entry)
        
    def log_influence(
        self,
        iteration: int,
        node: Any,
        influences: Dict[str, float],
        total_weight: float,
        old_state: Dict[str, float],
        new_state: Dict[str, float]
    ):
        """
        Log the influence each neighbor had on a node's update.
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        node : Any
            Node being updated
        influences : Dict[str, float]
            Dict mapping neighbor to their influence weight
        total_weight : float
            Sum of all neighbor weights
        old_state : Dict[str, float]
            Node state before update
        new_state : Dict[str, float]
            Node state after update
        """
        if not self.log_influence:
            return
            
        if self.sample_nodes is not None and node not in self.sample_nodes:
            return
            
        log_entry = {
            'iteration': iteration,
            'node': str(node),
            'n_neighbors': len(influences),
            'total_weight': total_weight,
            'neighbor_influences': {str(k): v for k, v in influences.items()},
            'old_emission': old_state.get('total emission from food', None),
            'new_emission': new_state.get('total emission from food', None),
            'emission_change': (new_state.get('total emission from food', 0) - 
                               old_state.get('total emission from food', 0)),
            'old_spending': old_state.get('total spending on food', None),
            'new_spending': new_state.get('total spending on food', None),
        }
        self.influence_logs.append(log_entry)
        
    def save_logs(self, prefix: str = ""):
        """
        Save all accumulated logs to disk.
        
        Parameters
        ----------
        prefix : str
            Prefix for output filenames
        """
        timestamp = self.run_id
        prefix = f"{prefix}_" if prefix else ""
        
        # Save metadata
        metadata_file = self.output_dir / f"{prefix}metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        print(f"Saved metadata to {metadata_file}")
        
        # Save weight logs
        if self.weight_logs:
            self._save_data(
                self.weight_logs,
                self.output_dir / f"{prefix}weight_logs_{timestamp}",
                "Weight logs"
            )
            
        # Save node state logs
        if self.node_state_logs:
            self._save_data(
                self.node_state_logs,
                self.output_dir / f"{prefix}node_state_logs_{timestamp}",
                "Node state logs"
            )
            
        # Save iteration logs
        if self.iteration_logs:
            self._save_data(
                self.iteration_logs,
                self.output_dir / f"{prefix}iteration_logs_{timestamp}",
                "Iteration logs"
            )
            
        # Save influence logs
        if self.influence_logs:
            self._save_data(
                self.influence_logs,
                self.output_dir / f"{prefix}influence_logs_{timestamp}",
                "Influence logs"
            )
            
        print(f"All logs saved to {self.output_dir}")
        
    def _save_data(self, data: List[Dict], filepath: Path, description: str):
        """Helper to save data in the specified format"""
        if self.save_format == "json":
            with open(f"{filepath}.json", 'w') as f:
                json.dump(data, f, indent=2, default=str)
            print(f"Saved {description} to {filepath}.json ({len(data)} entries)")
            
        elif self.save_format == "csv":
            df = pd.DataFrame(data)
            df.to_csv(f"{filepath}.csv", index=False)
            print(f"Saved {description} to {filepath}.csv ({len(data)} entries)")
            
        elif self.save_format == "pickle":
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved {description} to {filepath}.pkl ({len(data)} entries)")
            
    def get_weight_summary(self) -> pd.DataFrame:
        """Get summary statistics of weight calculations"""
        if not self.weight_logs:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.weight_logs)
        
        summary = df.groupby(['same_cluster']).agg({
            'spending_similarity': ['mean', 'std', 'min', 'max'],
            'household_similarity': ['mean', 'std', 'min', 'max'],
            'redmeat_similarity': ['mean', 'std', 'min', 'max'],
            'cluster_similarity': ['mean', 'std', 'min', 'max'],
            'base_weight': ['mean', 'std', 'min', 'max'],
            'final_weight': ['mean', 'std', 'min', 'max'],
        })
        
        return summary
        
    def get_iteration_summary(self) -> pd.DataFrame:
        """Get summary of iteration statistics"""
        if not self.iteration_logs:
            return pd.DataFrame()
        
        return pd.DataFrame(self.iteration_logs)
        
    def clear_logs(self):
        """Clear all accumulated logs (useful for memory management)"""
        self.weight_logs = []
        self.node_state_logs = []
        self.iteration_logs = []
        self.influence_logs = []
        
        
class LightweightLogger:
    """
    Lightweight logger that only tracks essential statistics.
    Use this for production runs when detailed logs would be too large.
    """
    
    def __init__(self, output_dir: str = "simulation_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.iteration_stats = []
        self.weight_stats = []
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
    def log_iteration_summary(
        self, 
        iteration: int, 
        emission_mean: float,
        emission_std: float,
        n_updates: int,
        n_skipped: int
    ):
        """Log only summary statistics per iteration"""
        self.iteration_stats.append({
            'iteration': iteration,
            'emission_mean': emission_mean,
            'emission_std': emission_std,
            'n_updates': n_updates,
            'n_skipped': n_skipped
        })
        
    def log_weight_summary(
        self,
        iteration: int,
        avg_weight: float,
        avg_cluster_sim: float,
        pct_same_cluster: float
    ):
        """Log only weight summary statistics"""
        self.weight_stats.append({
            'iteration': iteration,
            'avg_weight': avg_weight,
            'avg_cluster_sim': avg_cluster_sim,
            'pct_same_cluster': pct_same_cluster
        })
        
    def save_logs(self, prefix: str = ""):
        """Save lightweight logs"""
        timestamp = self.run_id
        prefix = f"{prefix}_" if prefix else ""
        
        if self.iteration_stats:
            df = pd.DataFrame(self.iteration_stats)
            filepath = self.output_dir / f"{prefix}iteration_summary_{timestamp}.csv"
            df.to_csv(filepath, index=False)
            print(f"Saved iteration summary to {filepath}")
            
        if self.weight_stats:
            df = pd.DataFrame(self.weight_stats)
            filepath = self.output_dir / f"{prefix}weight_summary_{timestamp}.csv"
            df.to_csv(filepath, index=False)
            print(f"Saved weight summary to {filepath}")

