"""
StreamingLogger.py

Streaming logger that writes directly to disk (CSV + structured JSON) to avoid memory overflow.
All entries are persisted immediately so logs survive mid-run interruptions.
Uses Python's standard logging module with JSON formatting for industry-standard structured logging.
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import networkx as nx

try:
    from .structured_logger import setup_json_logger, log_structured
except ImportError:
    from structured_logger import setup_json_logger, log_structured
import logging


class StreamingLogger:
    """
    Streaming logger that writes CSV files and mirrors every record to structured
    JSON-lines files using Python's standard logging module for richer diagnostics.
    """
    
    def __init__(
        self,
        output_dir: str = "simulation_logs",
        log_weights: bool = True,
        log_node_states: bool = False,  # Disabled by default (too verbose)
        log_iterations: bool = True,
        log_influence: bool = False,  # Disabled by default (too verbose)
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_weights = log_weights
        self.log_node_states = log_node_states
        self.log_iterations = log_iterations
        self.log_influence = log_influence
        
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.metadata = {}
        
        # File handles and CSV writers
        self.weight_file = None
        self.weight_writer = None
        self.node_state_file = None
        self.node_state_writer = None
        self.iteration_file = None
        self.iteration_writer = None
        self.influence_file = None
        self.influence_writer = None
        
        # Structured JSON loggers using Python's standard logging
        self.weight_json_logger: Optional[logging.Logger] = None
        self.node_state_json_logger: Optional[logging.Logger] = None
        self.iteration_json_logger: Optional[logging.Logger] = None
        self.influence_json_logger: Optional[logging.Logger] = None
        self.metadata_json_logger: Optional[logging.Logger] = None
        
        self._weight_row_counter = 0
        
        self._initialize_files()
        
    def _initialize_files(self):
        """Open file handles and create CSV writers + structured JSON loggers."""
        if self.log_weights:
            weight_path = self.output_dir / f"weight_logs_{self.run_id}.csv"
            self.weight_file = open(weight_path, 'w', newline='', buffering=8192)
            self.weight_writer = None
            self.weight_json_logger = setup_json_logger(
                name=f"simulation.weights.{self.run_id}",
                log_file=self.output_dir / f"weight_logs_{self.run_id}.jsonl",
                level=logging.INFO
            )
        
        if self.log_node_states:
            node_state_path = self.output_dir / f"node_state_logs_{self.run_id}.csv"
            self.node_state_file = open(node_state_path, 'w', newline='', buffering=8192)
            self.node_state_writer = None
            self.node_state_json_logger = setup_json_logger(
                name=f"simulation.node_states.{self.run_id}",
                log_file=self.output_dir / f"node_state_logs_{self.run_id}.jsonl",
                level=logging.INFO
            )
        
        if self.log_iterations:
            iteration_path = self.output_dir / f"iteration_logs_{self.run_id}.csv"
            self.iteration_file = open(iteration_path, 'w', newline='', buffering=8192)
            self.iteration_writer = None
            self.iteration_json_logger = setup_json_logger(
                name=f"simulation.iterations.{self.run_id}",
                log_file=self.output_dir / f"iteration_logs_{self.run_id}.jsonl",
                level=logging.INFO
            )
        
        if self.log_influence:
            influence_path = self.output_dir / f"influence_logs_{self.run_id}.csv"
            self.influence_file = open(influence_path, 'w', newline='', buffering=8192)
            self.influence_writer = None
            self.influence_json_logger = setup_json_logger(
                name=f"simulation.influence.{self.run_id}",
                log_file=self.output_dir / f"influence_logs_{self.run_id}.jsonl",
                level=logging.INFO
            )
        
        self.metadata_json_logger = setup_json_logger(
            name=f"simulation.metadata.{self.run_id}",
            log_file=self.output_dir / f"metadata_{self.run_id}.jsonl",
            level=logging.INFO
        )
        
    def set_metadata(self, **kwargs):
        self.metadata.update(kwargs)
        self.metadata['run_id'] = self.run_id
        self.metadata['timestamp'] = datetime.now().isoformat()
        
        metadata_path = self.output_dir / f"metadata_{self.run_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        if self.metadata_json_logger:
            log_structured(self.metadata_json_logger, logging.INFO, self.metadata)
            
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
        neighbor_cluster: Any = None,
        node_attrs: Optional[Dict[str, Any]] = None,
        neighbor_attrs: Optional[Dict[str, Any]] = None
    ):
        if not self.log_weights:
            return
            
        if self.weight_writer is None:
            fieldnames = [
                'iteration', 'node', 'neighbor', 'edge_id',
                'spending_similarity', 'household_similarity', 'redmeat_similarity', 'cluster_similarity',
                'alpha', 'beta', 
                'component1', 'component2', 'final_weight',
                'node_cluster', 'neighbor_cluster', 'same_cluster',
                'node_adults', 'neighbor_adults',
                'node_total_spending_on_food', 'neighbor_total_spending_on_food',
                'node_redmeat', 'neighbor_redmeat',
                'node_cult1', 'neighbor_cult1',
                'node_income', 'neighbor_income',
                'node_HC24', 'neighbor_HC24',
                'node_cult', 'neighbor_cult',
                'node_consumption_rate', 'neighbor_consumption_rate',
                'node_dur_spend_r', 'neighbor_dur_spend_r',
                'node_ndur_spend_r', 'neighbor_ndur_spend_r',
                'node_serv_spend_r', 'neighbor_serv_spend_r'
            ]
            self.weight_writer = csv.DictWriter(self.weight_file, fieldnames=fieldnames)
            self.weight_writer.writeheader()
            self.weight_file.flush()
            
        # Calculate the two components separately
        component1 = alpha * spending_sim * household_sim
        component2 = beta * redmeat_sim * cluster_sim
        
        row = {
            'iteration': iteration,
            'node': str(node),
            'neighbor': str(neighbor),
            'edge_id': f"{node}->{neighbor}",
            'spending_similarity': spending_sim,
            'household_similarity': household_sim,
            'redmeat_similarity': redmeat_sim,
            'cluster_similarity': cluster_sim,
            'alpha': alpha,
            'beta': beta,
            'component1': component1,
            'component2': component2,
            'final_weight': final_weight,
            'node_cluster': str(node_cluster) if node_cluster is not None else '',
            'neighbor_cluster': str(neighbor_cluster) if neighbor_cluster is not None else '',
            'same_cluster': (node_cluster == neighbor_cluster) if (node_cluster is not None and neighbor_cluster is not None) else ''
        }
        
        attr_keys = [
            ('adults', 'node_adults', 'neighbor_adults'),
            ('total spending on food', 'node_total_spending_on_food', 'neighbor_total_spending_on_food'),
            ('redmeat', 'node_redmeat', 'neighbor_redmeat'),
            ('cult1', 'node_cult1', 'neighbor_cult1'),
            ('income', 'node_income', 'neighbor_income'),
            ('HC24', 'node_HC24', 'neighbor_HC24'),
            ('cult', 'node_cult', 'neighbor_cult'),
            ('consumption_rate', 'node_consumption_rate', 'neighbor_consumption_rate'),
            ('dur_spend_r', 'node_dur_spend_r', 'neighbor_dur_spend_r'),
            ('ndur_spend_r', 'node_ndur_spend_r', 'neighbor_ndur_spend_r'),
            ('serv_spend_r', 'node_serv_spend_r', 'neighbor_serv_spend_r'),
        ]
        node_attrs = node_attrs or {}
        neighbor_attrs = neighbor_attrs or {}
        for key, node_field, neighbor_field in attr_keys:
            row[node_field] = node_attrs.get(key, '')
            row[neighbor_field] = neighbor_attrs.get(key, '')
        
        self.weight_writer.writerow(row)
        self._weight_row_counter += 1
        if self._weight_row_counter % 100 == 0:
            self.weight_file.flush()
        
        if self.weight_json_logger:
            log_structured(self.weight_json_logger, logging.INFO, row)
            
    def log_node_state(
        self,
        iteration: int,
        node: Any,
        state: Dict[str, float],
        stage: str = "after_update"
    ):
        if not self.log_node_states:
            return
            
        if self.node_state_writer is None:
            fieldnames = ['iteration', 'node', 'stage'] + list(state.keys())
            self.node_state_writer = csv.DictWriter(self.node_state_file, fieldnames=fieldnames)
            self.node_state_writer.writeheader()
            self.node_state_file.flush()
            
        row = {
            'iteration': iteration,
            'node': str(node),
            'stage': stage,
            **{k: v for k, v in state.items() if isinstance(v, (int, float, str, bool))}
        }
        self.node_state_writer.writerow(row)
        self.node_state_file.flush()
        
        if self.node_state_json_logger:
            log_structured(self.node_state_json_logger, logging.INFO, row)
            
    def log_iteration_stats(
        self,
        iteration: int,
        G: nx.Graph,
        static_nodes: List = None
    ):
        if not self.log_iterations:
            return
            
        static_nodes = static_nodes or []
        dynamic_nodes = [n for n in G.nodes() if n not in static_nodes]
        
        import numpy as np
        
        def get_stats(nodes):
            if not nodes:
                return {
                    'n_nodes': 0,
                    'emission_mean': 0.0,
                    'emission_std': 0.0,
                    'emission_min': 0.0,
                    'emission_max': 0.0,
                    'spending_mean': 0.0,
                    'spending_std': 0.0,
                    'redmeat_mean': 0.0,
                    'redmeat_std': 0.0,
                }
            emissions = [G.nodes[n].get('total emission from food', 0) for n in nodes]
            spending = [G.nodes[n].get('total spending on food', 0) for n in nodes]
            redmeat = [G.nodes[n].get('redmeat', 0) for n in nodes]
            return {
                'n_nodes': len(nodes),
                'emission_mean': float(np.mean(emissions)),
                'emission_std': float(np.std(emissions)),
                'emission_min': float(np.min(emissions)),
                'emission_max': float(np.max(emissions)),
                'spending_mean': float(np.mean(spending)),
                'spending_std': float(np.std(spending)),
                'redmeat_mean': float(np.mean(redmeat)),
                'redmeat_std': float(np.std(redmeat)),
            }
        
        if self.iteration_writer is None:
            fieldnames = [
                'iteration', 'total_nodes', 'total_edges', 'n_static', 'n_dynamic',
                'dynamic_n_nodes', 'dynamic_emission_mean', 'dynamic_emission_std',
                'dynamic_emission_min', 'dynamic_emission_max',
                'dynamic_spending_mean', 'dynamic_spending_std',
                'dynamic_redmeat_mean', 'dynamic_redmeat_std',
                'static_n_nodes', 'static_emission_mean', 'static_emission_std',
                'static_emission_min', 'static_emission_max',
                'static_spending_mean', 'static_spending_std',
                'static_redmeat_mean', 'static_redmeat_std',
            ]
            self.iteration_writer = csv.DictWriter(self.iteration_file, fieldnames=fieldnames)
            self.iteration_writer.writeheader()
            self.iteration_file.flush()
            
        dynamic_stats = get_stats(dynamic_nodes)
        static_stats = get_stats(static_nodes)
        row = {
            'iteration': iteration,
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'n_static': len(static_nodes),
            'n_dynamic': len(dynamic_nodes),
            **{f'dynamic_{k}': v for k, v in dynamic_stats.items()},
            **{f'static_{k}': v for k, v in static_stats.items()},
        }
        self.iteration_writer.writerow(row)
        self.iteration_file.flush()
        
        if self.iteration_json_logger:
            log_structured(self.iteration_json_logger, logging.INFO, row)
            
    def log_influence(
        self,
        iteration: int,
        node: Any,
        influences: Dict[str, float],
        total_weight: float,
        old_state: Dict[str, float],
        new_state: Dict[str, float]
    ):
        if not self.log_influence:
            return
            
        if self.influence_writer is None:
            fieldnames = [
                'iteration', 'node', 'n_neighbors', 'total_weight',
                'neighbor_influences', 'old_emission', 'new_emission', 'emission_change',
                'old_spending', 'new_spending'
            ]
            self.influence_writer = csv.DictWriter(self.influence_file, fieldnames=fieldnames)
            self.influence_writer.writeheader()
            self.influence_file.flush()
            
        row = {
            'iteration': iteration,
            'node': str(node),
            'n_neighbors': len(influences),
            'total_weight': total_weight,
            'neighbor_influences': json.dumps({str(k): v for k, v in influences.items()}),
            'old_emission': old_state.get('total emission from food', None),
            'new_emission': new_state.get('total emission from food', None),
            'emission_change': (new_state.get('total emission from food', 0) -
                               old_state.get('total emission from food', 0)),
            'old_spending': old_state.get('total spending on food', None),
            'new_spending': new_state.get('total spending on food', None),
        }
        self.influence_writer.writerow(row)
        self.influence_file.flush()
        
        if self.influence_json_logger:
            log_structured(self.influence_json_logger, logging.INFO, row)
            
    def close(self):
        if self.weight_file:
            self.weight_file.flush()
            self.weight_file.close()
        if self.node_state_file:
            self.node_state_file.flush()
            self.node_state_file.close()
        if self.iteration_file:
            self.iteration_file.flush()
            self.iteration_file.close()
        if self.influence_file:
            self.influence_file.flush()
            self.influence_file.close()
        
        # Standard logging module handles cleanup automatically via logging.shutdown()
        # Force flush all handlers
        for logger in [self.weight_json_logger, self.node_state_json_logger, 
                      self.iteration_json_logger, self.influence_json_logger,
                      self.metadata_json_logger]:
            if logger:
                for handler in logger.handlers:
                    handler.flush()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

