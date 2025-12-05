import networkx as nx
from typing import Dict, Optional, List, Union, Any, Callable
import math
import random
import numpy as np

class NodeSelector:
    @staticmethod
    def get_nodes_by_quantiles(
        G: nx.Graph, 
        x: float, 
        variable: Union[str, List[str]],
        use_random: bool = False
        ) -> Dict[str, List]:
        """
        Splits the nodes of graph G into 10 quantiles based on the values of the specified node attribute(s) 'variable'.
        For each quantile group, selects the top x fraction of nodes with the highest degree (or randomly if use_random is True).
        Handling an empty variable list:
        If variable is an empty list, the method ignores quantiles and simply selects the top x-fraction of nodes (by degree) 
        from the entire graph (or a random sample of that size if use_random=True). And return a list (not dict in this case)
        Parameters:
            G (nx.Graph): The input graph.
            x (float): Fraction of nodes to select within each quantile (0 < x <= 1).
            variable (Union[str, List[str]]): The node attribute(s) used to compute quantiles. If a list is provided, the final value is computed as the sum of the attributes in the list. If the list is empty, the function returns the top high degree nodes across the entire graph based on degree.
            use_random (bool): If True, selects nodes randomly instead of by degree.
            
        Returns:
            Dict[str, List]: A dictionary where each key is a quantile label ("Q1" through "Q10") and the value is a list 
                            of selected node identifiers from that quantile.
        """
        # Determine attribute values for nodes.
        if isinstance(variable, list):
            if not variable:  # If the list is empty, return high degree nodes from the entire graph.
                all_nodes = list(G.nodes())
                target_size = math.ceil(len(all_nodes) * x)
                if use_random:
                    return random.sample(all_nodes, target_size)
                else:
                    sorted_nodes = sorted(all_nodes, key=lambda n: G.degree(n), reverse=True)
                    return sorted_nodes[:target_size]
            else:
                nodes_with_val = []
                for node in G.nodes():
                    # Only consider nodes that have all specified attributes
                    if all(v in G.nodes[node] for v in variable):
                        # Sum the values of the attributes
                        total = sum(G.nodes[node][v] for v in variable)
                        nodes_with_val.append((node, total))
        else:
            nodes_with_val = []
            for node in G.nodes():
                if variable not in G.nodes[node]:
                    continue
                val = G.nodes[node][variable]
                nodes_with_val.append((node, val))
        
        if not nodes_with_val:
            print("No nodes available with the specified attribute.")
            return {}
        
        # Step 2: Compute quantile edges (0th, 10th, ..., 100th percentiles) using the 'variable' values.
        values = np.array([v for _, v in nodes_with_val])
        quantile_edges = np.percentile(values, np.arange(0, 101, 10))
        
        # Initialize quantile groups Q1 through Q10.
        quantile_groups = {f"Q{i+1}": [] for i in range(10)}
        
        # Step 3: Assign each node to the appropriate quantile.
        # For quantiles 1-9: [edge[i], edge[i+1]), for the 10th quantile: [edge[9], edge[10]].
        for node, val in nodes_with_val:
            quantile_index = None
            for i in range(10):
                if i < 9:
                    if quantile_edges[i] <= val < quantile_edges[i+1]:
                        quantile_index = i
                        break
                else:
                    if quantile_edges[i] <= val <= quantile_edges[i+1]:
                        quantile_index = i
                        break
            if quantile_index is not None:
                quantile_groups[f"Q{quantile_index+1}"].append(node)
        
        # Step 4: For each quantile group, select the top x fraction of nodes based on degree (or randomly).
        result = {}
        for q_label, nodes in quantile_groups.items():
            if not nodes:
                result[q_label] = []
                continue
            # Calculate the number of nodes to select in this quantile.
            target_size = min(math.ceil(len(nodes) * x), len(nodes))
            
            if use_random:
                selected = random.sample(nodes, target_size)
            else:
                # Sort nodes in the current quantile by degree (descending order).
                degrees = {node: G.degree(node) for node in nodes}
                sorted_nodes = sorted(degrees.items(), key=lambda item: item[1], reverse=True)
                selected = [node for node, _ in sorted_nodes][:target_size]
            
            result[q_label] = selected

        return result

    @staticmethod
    def get_nodes_by_category(
        G: nx.Graph,
        x: float,
        variable: str,
        use_random: bool = False
    ) -> Dict[Any, List]:
        """
        Groups nodes of graph G by the distinct values of the specified categorical attribute 'variable'.
        For each group, selects the top x fraction of nodes with the highest degree (or randomly if use_random is True).
        
        Parameters:
            G (nx.Graph): The input graph.
            x (float): Fraction of nodes to select within each group (0 < x <= 1).
            variable (str): The categorical node attribute used for grouping.
            use_random (bool): If True, selects nodes randomly instead of by degree.
        
        Returns:
            Dict[Any, List]: A dictionary where each key is a categorical value and the value is a list of selected node identifiers.
        """
        # Group nodes by the categorical attribute value.
        groups = {}
        for node in G.nodes():
            if variable not in G.nodes[node]:
                continue
            attr_val = G.nodes[node][variable]
            # If the attribute value is a float, convert it to binary: 0 if value is 0, otherwise 1.
            if isinstance(attr_val, float):
                cat_value = 0 if attr_val == 0.0 else 1
            else:
                cat_value = attr_val
            groups.setdefault(cat_value, []).append(node)
        
        result = {}
        for cat_value, nodes in groups.items():
            target_size = min(math.ceil(len(nodes) * x), len(nodes))
            if use_random:
                selected = random.sample(nodes, target_size)
            else:
                degrees = {node: G.degree(node) for node in nodes}
                sorted_nodes = sorted(degrees.items(), key=lambda item: item[1], reverse=True)
                selected = [node for node, _ in sorted_nodes][:target_size]
            result[cat_value] = selected
        
        return result
    
    @staticmethod
    def get_nodes_by_quantiles_centrality(
        G: nx.Graph, 
        x: float, 
        variable: Union[str, List[str]],
        use_random: bool = False,
        # Accept an optional centrality function. If not provided, defaults to closeness centrality.
        centrality_measure: Optional[Callable[[nx.Graph], Dict[Any, float]]] = None
    ) -> Dict[str, List]:
        """
        Splits the nodes of graph G into 10 quantiles based on the values of the specified node attribute(s) 'variable'.
        For each quantile group, selects the top x fraction of nodes with the highest closeness centrality 
        (or randomly if use_random is True).

        Parameters:
            G (nx.Graph): The input graph.
            x (float): Fraction of nodes to select within each quantile (0 < x <= 1).
            variable (Union[str, List[str]]): The node attribute(s) used to compute quantiles.
            use_random (bool): If True, selects nodes randomly instead of by centrality.
            centrality_measure (Optional[Callable]): A function that computes a centrality measure given a graph.
                                                     If None, uses closeness centrality by default.

        Returns:
            Dict[str, List]: A dictionary where each key is a quantile label ("Q1" through "Q10") and the value 
                             is a list of selected node identifiers from that quantile.
        """
        # Compute the centrality measure. Default to closeness centrality if none is provided.
        if centrality_measure is None:
            centrality = nx.closeness_centrality(G)
        else:
            centrality = centrality_measure(G)
        
        # Step 1: Determine attribute values for nodes.
        if isinstance(variable, list):
            if not variable:  # If empty list, return top nodes based on centrality from the entire graph.
                all_nodes = list(G.nodes())
                target_size = math.ceil(len(all_nodes) * x)
                if use_random:
                    return {"Top":random.sample(all_nodes, target_size)}
                else:
                    sorted_nodes = sorted(all_nodes, key=lambda n: centrality[n], reverse=True)
                    return { "Top": sorted_nodes[:target_size] }
            else:
                nodes_with_val = []
                for node in G.nodes():
                    # Only consider nodes that have all specified attributes.
                    if all(v in G.nodes[node] for v in variable):
                        total = sum(G.nodes[node][v] for v in variable)
                        nodes_with_val.append((node, total))
        else:
            nodes_with_val = []
            for node in G.nodes():
                if variable not in G.nodes[node]:
                    continue
                val = G.nodes[node][variable]
                nodes_with_val.append((node, val))
        
        if not nodes_with_val:
            print("No nodes available with the specified attribute.")
            return {}
        
        # Step 2: Compute quantile edges (0th, 10th, ..., 100th percentiles) using the attribute values.
        values = np.array([v for _, v in nodes_with_val])
        quantile_edges = np.percentile(values, np.arange(0, 101, 10))
        
        # Initialize quantile groups Q1 through Q10.
        quantile_groups = {f"Q{i+1}": [] for i in range(10)}
        
        # Step 3: Assign each node to the appropriate quantile.
        for node, val in nodes_with_val:
            quantile_index = None
            for i in range(10):
                if i < 9:
                    if quantile_edges[i] <= val < quantile_edges[i+1]:
                        quantile_index = i
                        break
                else:
                    if quantile_edges[i] <= val <= quantile_edges[i+1]:
                        quantile_index = i
                        break
            if quantile_index is not None:
                quantile_groups[f"Q{quantile_index+1}"].append(node)
        
        # Step 4: For each quantile group, select the top x fraction of nodes based on closeness centrality (or randomly).
        result = {}
        for q_label, nodes in quantile_groups.items():
            if not nodes:
                result[q_label] = []
                continue
            target_size = min(math.ceil(len(nodes) * x), len(nodes))
            
            if use_random:
                selected = random.sample(nodes, target_size)
            else:
                sorted_nodes = sorted(nodes, key=lambda n: centrality[n], reverse=True)
                selected = sorted_nodes[:target_size]
            
            result[q_label] = selected

        return result
    

    @staticmethod
    def get_nodes_by_any_quantiles(
        G: nx.Graph, 
        x: float, 
        variable: Union[str, List[str]],
        num_quantiles: int = 10,
        use_random: bool = False
        ) -> Union[Dict[str, List[Any]], List[Any]]:
        """
        Splits the nodes of graph G into num_quantiles quantile groups based on the values of the specified node attribute(s).
        For each quantile group, selects the top x fraction of nodes with the highest degree (or randomly if use_random is True).
        Handling an empty variable list:
        If variable is an empty list, the method ignores quantiles and simply selects the top x-fraction of nodes (by degree) 
        from the entire graph (or a random sample of that size if use_random=True). Returns a list of selected nodes in this case.

        Parameters:
            G (nx.Graph): The input graph.
            x (float): Fraction of nodes to select within each quantile (0 < x <= 1).
            variable (Union[str, List[str]]): The node attribute(s) used to compute quantiles. If a list is provided, the final value is computed as the sum of the attributes in the list. If the list is empty, the function returns the top high degree nodes across the entire graph based on degree.
            num_quantiles (int): Number of quantile groups to split nodes into.
            use_random (bool): If True, selects nodes randomly instead of by degree.
        
        Returns:
            Union[Dict[str, List[Any]], List[Any]]: If variable is non-empty, returns a dictionary where each key is a quantile label ("Q1" through "Q{num_quantiles}") and the value is a list of selected node identifiers from that quantile. If variable is an empty list, returns a list of selected node identifiers.
        """
        # Determine attribute values for nodes.
        if isinstance(variable, list) and not variable:
            # Empty list: select top x fraction based on degree across the entire graph
            all_nodes = list(G.nodes())
            target_size = math.ceil(len(all_nodes) * x)
            if use_random:
                return random.sample(all_nodes, target_size)
            else:
                sorted_nodes = sorted(all_nodes, key=lambda n: G.degree(n), reverse=True)
                return sorted_nodes[:target_size]

        # Gather nodes and their attribute-based values
        nodes_with_val = []
        if isinstance(variable, list):
            for node in G.nodes():
                if all(attr in G.nodes[node] for attr in variable):
                    total = sum(G.nodes[node][attr] for attr in variable)
                    nodes_with_val.append((node, total))
        else:
            for node in G.nodes():
                if variable in G.nodes[node]:
                    nodes_with_val.append((node, G.nodes[node][variable]))

        if not nodes_with_val:
            print("No nodes available with the specified attribute.")
            return {}

        # Compute quantile edges
        values = np.array([val for _, val in nodes_with_val])
        # Percentile edges from 0 to 100 in num_quantiles+1 steps
        percentiles = np.linspace(0, 100, num_quantiles + 1)
        quantile_edges = np.percentile(values, percentiles)

        # Initialize quantile groups
        quantile_groups: Dict[str, List[Any]] = {f"Q{i+1}": [] for i in range(num_quantiles)}

        # Assign nodes to quantiles
        for node, val in nodes_with_val:
            for i in range(num_quantiles):
                low = quantile_edges[i]
                high = quantile_edges[i+1]
                # Inclusive on the lower bound for all, inclusive on upper for last quantile
                if (i < num_quantiles - 1 and low <= val < high) or (i == num_quantiles - 1 and low <= val <= high):
                    quantile_groups[f"Q{i+1}"].append(node)
                    break

        # Select top x fraction from each quantile group
        result: Dict[str, List[Any]] = {}
        for q_label, nodes in quantile_groups.items():
            if not nodes:
                result[q_label] = []
                continue
            target_size = min(math.ceil(len(nodes) * x), len(nodes))
            if use_random:
                selected = random.sample(nodes, target_size)
            else:
                sorted_by_degree = sorted(nodes, key=lambda n: G.degree(n), reverse=True)
                selected = sorted_by_degree[:target_size]
            result[q_label] = selected

        return result


