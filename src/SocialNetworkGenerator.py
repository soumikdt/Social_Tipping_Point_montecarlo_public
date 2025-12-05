import random
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from typing import Callable
import os, time, multiprocessing as mp
from networkx.algorithms.swap import double_edge_swap
from datetime import datetime
import igraph as ig
from typing import Optional

timestamp = datetime.now().strftime("%m%d_%H%M%S")
ts = timestamp



class SocialNetworkAnalyzer:
    _SocialNetworkAnalyzerCounter = 1
    """
    A class providing methods for creating and analyzing social networks 
    using various graph-generation and analysis algorithms.
    """
    def __init__(self, dataframe: pd.DataFrame, col = 'cult1', col2 = 'income', col3 = 'HC24', col4 = 'cult', seed = 1996):
        """
        Initialize the SocialNetworkAnalyzer with a Pandas DataFrame.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            A DataFrame containing node attributes. 
            Should include columns like 'cult', 'HC24', etc., depending on use.
        """
        self.dataframe = dataframe
        self.id = SocialNetworkAnalyzer._SocialNetworkAnalyzerCounter 
        SocialNetworkAnalyzer._SocialNetworkAnalyzerCounter += 1
        self.graph :nx.Graph = None
        self.col = col
        self.col2 = col2
        self._col3 = col3
        self._col4 = col4
        self.plot_prefix: str = '' 
        random.seed(seed)
        np.random.seed(seed)

    def __repr__(self):
        return f"(SocialNetworkAnalyzer: {self.id})"

    ############################################################################
    # =========================== GRAPH GENERATORS ============================ #
    ############################################################################
    
    def havel_hakimi_graph_modified_with_attribute(self, df: pd.DataFrame, func : Callable, seed=None) -> nx.Graph:
        """
        Constructs a graph using the Havel–Hakimi algorithm with a degree sequence 
        derived from the attribute values in the DataFrame column using the transformation function as input.
        Each node's computed degree is capped at (n - 2) (no node can exceed the max possible degree).
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing node attributes.
        col : str
            Column name from which to compute the degree (e.g., 'cult1').
        seed : integer, random_state, or None (default)
            Random seed or random state (unused in this function).

        Raises
        ------
        nx.NetworkXError
            If the computed degree sequence is not graphical.

        Returns
        -------
        G : networkx.Graph
            A graph constructed using the Havel–Hakimi algorithm with node attributes assigned.
        """
        n = df.shape[0]
        col = self.col
        col2 = self.col2 
        # Sort the DataFrame by the specified column in descending order.
        df_sorted = df.sort_values(by=col, ascending=False).reset_index(drop=True)
        
        # Compute the degree sequence using:  degree = 3 * log(x+1) + 2
        # and clamp each degree to (n-1).
        degree_sequence = []
        for indx, (val, val2) in enumerate(zip(df_sorted[col], df_sorted[col2])):
            if (indx + 1) % 100_000 == 0 or indx + 1 == n:
                print(f"[HH] computing degrees: {indx+1}/{n} rows processed")
            raw_degree = func(val, val2) 
            d = int(round(raw_degree))
            d = min(d, n - 2)  # clamp
            degree_sequence.append(d)
        

        # Ensure that the sum of the degree sequence is even (needed for a simple undirected graph).
        if sum(degree_sequence) % 2 != 0:
            degree_sequence[-1] += 1

        #print("Computed degree sequence:", degree_sequence)
        #print("Sum of degrees:", sum(degree_sequence))
        #print("Graphical check (Erdős–Gallai):", nx.is_valid_degree_sequence_erdos_gallai(degree_sequence))

        # Check if the sequence is graphical.
        if not nx.is_graphical(degree_sequence):
            raise nx.NetworkXError("The computed degree sequence is not graphical.")

        # Create the graph using the Havel–Hakimi algorithm.
        G = nx.havel_hakimi_graph(degree_sequence)

        # Assign node attributes from the sorted DataFrame.
        for i, node in enumerate(G.nodes()):
            G.nodes[node].update(df_sorted.iloc[i].to_dict())
        
        return G

    @staticmethod
    def powerlaw_cluster_graph_modified_with_attribute_and_maxdegree2(n: int,
                                                                      m: int,
                                                                      p: float,
                                                                      deg: int,
                                                                      df: pd.DataFrame,
                                                                      col: str,
                                                                      seed=1996) -> nx.Graph:
        """
        Holme and Kim algorithm for growing graphs with a power-law degree distribution and 
        approximate average clustering, with additional attributes and a degree constraint.
        
        Parameters
        ----------
        n : int
            The number of nodes.
        m : int
            The number of edges to attach from a new node to existing nodes.
        p : float
            Probability of adding a triangle after adding a random edge.
        deg : int
            Maximum degree allowed for any node.
        df : pandas.DataFrame
            DataFrame containing node attributes.
        col : str
            Column name to sort the DataFrame by before constructing the graph.
        seed : integer, random.Random, or None
            Random seed or random state.

        Raises
        ------
        NetworkXError
            If arguments are out of acceptable ranges or if sizes mismatch.

        Returns
        -------
        G : networkx.Graph
        """
        if m < 1 or n < m:
            raise nx.NetworkXError(f"NetworkXError must have m>1 and m<n, m={m},n={n}")
        if p > 1 or p < 0:
            raise nx.NetworkXError(f"NetworkXError p must be in [0,1], p={p}")
        if n != df.shape[0]:
            raise nx.NetworkXError(f"n and dataframe length must be equal, n={n}, dataframe length={df.shape[0]}")

        if seed is None:
            seed = random

        # Sort DataFrame by the specified column in descending order
        df_sorted = df.sort_values(by=col, ascending=False).reset_index(drop=True)

        # Create initial graph with m nodes
        G = nx.empty_graph(m)
        repeated_nodes = list(G.nodes())
        source = m

        # Grow the graph
        while source < n:

            
            # Filter out nodes that have reached the degree cap
            repeated_nodes = [node for node in repeated_nodes if G.degree(node) < deg]

            # Attempt to pick m targets
            if len(repeated_nodes) < m:
                break  # cannot attach further if not enough valid nodes

            possible_targets = seed.sample(repeated_nodes, m)

            # Attach the new node to one target first
            target = possible_targets.pop()
            G.add_edge(source, target)
            repeated_nodes.append(target)
            count = 1

            while count < m:
                if seed.random() < p:
                    # Clustering step: add triangle
                    neighborhood = [
                        nbr for nbr in G.neighbors(target)
                        if not G.has_edge(source, nbr) and nbr != source and G.degree(nbr) < deg
                    ]
                    if neighborhood:
                        nbr = seed.choice(neighborhood)
                        G.add_edge(source, nbr)
                        repeated_nodes.append(nbr)
                        count += 1
                        continue

                # Otherwise, preferential attachment step
                if possible_targets:
                    target = possible_targets.pop()
                    G.add_edge(source, target)
                    repeated_nodes.append(target)
                    count += 1

            repeated_nodes.extend([source] * m)
            source += 1

        # Assign attributes based on degree rank
        sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
        for i, node in enumerate(sorted_nodes):
            G.nodes[node].update(df_sorted.iloc[i].to_dict())

        return G

    ############################################################################
    # ========================= EDGE REORIENTATION ============================ #
    ############################################################################

    @staticmethod
    def reorient_edges_preserve_degree(G: nx.Graph, p: float = 0.1) -> None:
        """
        Reorient p fraction of edges in the network while preserving node degrees.
        
        Parameters
        ----------
        G : networkx.Graph
            The graph whose edges will be reoriented.
        p : float
            Fraction of edges to reorient (rewire).
        """
        edges = list(G.edges())
        num_reorient = int(p * len(edges))
        swaps = 0
        while swaps < num_reorient:
            if swaps%5000==0:
                print(swaps)
            edge1, edge2 = random.sample(edges, 2)
            # Ensure the selected edges do not share a node
            if len(set(edge1 + edge2)) == 4:
                # Perform the edge swap
                u1, v1 = edge1
                u2, v2 = edge2
                if not G.has_edge(u1, v2) and not G.has_edge(u2, v1):
                    G.remove_edge(u1, v1)
                    G.remove_edge(u2, v2)
                    G.add_edge(u1, v2)
                    G.add_edge(u2, v1)
                    edges.remove(edge1)
                    edges.remove(edge2)
                    edges.append((u1, v2))
                    edges.append((u2, v1))
                    swaps += 1

    @staticmethod
    def reorient_edges_preserve_degree1(G: nx.Graph, p: float = 0.1) -> None:
        """
        Reorient p fraction of edges while preserving node degrees,
        using NetworkX’s built-in double_edge_swap.
        Prints the swap count at 0 and then at the end, matching the old output.
        """
        m = G.number_of_edges()
        nswap = int(p * m)
        if nswap == 0:
            print(0)
            return

        # print at the “0 swaps” mark
        print(0)
        t0 = time.perf_counter()

        # allow up to 10 tries per requested swap (same heuristic NX uses)
        double_edge_swap(
            G,
            nswap=nswap,
            max_tries=nswap * 10,
            seed=None,
            connected=False
        )

        # print the final swap count
        print(nswap)
        # optionally you can also print timing if you like:
        # print(f"  (completed in {time.perf_counter()-t0:.1f}s)")

    ################################################################################
    # helper – run inside a worker process
    ################################################################################
    @staticmethod
    def _parallel_swap_worker(args):
        """
        Parameters
        ----------
        sub_edges : list[(int,int)]
            Edges whose *both* endpoints live in this partition.
        p         : float
            Fraction of those edges to rewire.
        seed      : int
            Per–worker RNG seed.

        Returns
        -------
        tuple[ set[(int,int)], set[(int,int)] ]
            (edges_to_remove , edges_to_add)
        """
        sub_edges, p, seed = args
        subG = nx.Graph()
        subG.add_edges_from(sub_edges)

        nswap = int(p * len(sub_edges))
        if nswap:
            double_edge_swap(subG,
                            nswap=nswap,
                            max_tries=nswap*10,
                            seed=seed,
                            connected=False)

        new_edges = set(subG.edges())
        old_edges = set(sub_edges)
        return old_edges - new_edges, new_edges - old_edges      # deltas
    ################################################################################

    @staticmethod
    def reorient_edges_preserve_degree_parallel(G: nx.Graph,
                                                p: float = 0.10,
                                                processes = None,
                                                min_batch_size: int = 25_000) -> None:
        """
        Random-sized batching + parallel double-edge swaps.

        Parameters
        ----------
        G : nx.Graph               (modified in place)
        p : float                  fraction of edges to rewire
        processes : int | None     CPU cores to use (defaults to os.cpu_count())
        min_batch_size : int       minimum #edges in a partition
        """
        t0 = time.perf_counter()
        m  = G.number_of_edges()
        if m == 0 or p <= 0:
            print("[rewire] nothing to do")
            return

        # -----------------------------------------------------------------
        # 1) partition the edges into node-disjoint buckets
        # -----------------------------------------------------------------
        if processes is None:
            processes = os.cpu_count() or 1

        # hash node → bucket  (simple, fast, randomised each run)
        rnd = random.randrange(1 << 30)
        def bucket_of(node):        # xor with a random integer first
            return (hash(node) ^ rnd) % processes

        buckets = [[] for _ in range(processes)]
        for u, v in G.edges():
            # put edge only if *both* endpoints hash to the same bucket
            b = bucket_of(u)
            if b == bucket_of(v):
                buckets[b].append((u, v))
        # tiny partitions get merged
        buckets = [b for b in buckets if len(b) >= min_batch_size]
        if not buckets:                         # fall back to single-thread
            buckets = [list(G.edges())]

        # -----------------------------------------------------------------
        # 2) run double_edge_swap on each bucket in parallel
        # -----------------------------------------------------------------
        print(f"[rewire] {len(buckets)} disjoint batches, "
            f"{sum(len(b) for b in buckets):,}/{m:,} edges total")

        with mp.Pool(len(buckets)) as pool:
            deltas = pool.map(
                SocialNetworkAnalyzer._parallel_swap_worker,
                [(b, p, random.randrange(1 << 30)) for b in buckets]
            )

        # -----------------------------------------------------------------
        # 3) apply edge deltas back to the master graph
        # -----------------------------------------------------------------
        total_swaps = 0
        for to_remove, to_add in deltas:
            G.remove_edges_from(to_remove)
            G.add_edges_from(to_add)
            total_swaps += len(to_add)

        elapsed = time.perf_counter() - t0
        print(f"[rewire] finished {total_swaps:,} swaps in "
            f"{elapsed:,.1f}s  "
            f"({total_swaps/elapsed:,.0f} swaps/s)")
    ############################################################################
    # ========================= NETWORK STATISTICS ============================ #
    ############################################################################

    @staticmethod
    def compute_network_statistics1(graph: nx.Graph, attribute: str) -> dict:
        """
        Compute various network statistics: diameter, average shortest path length,
        clustering coefficient, and homophily (assortativity by an attribute).

        Parameters
        ----------
        graph : networkx.Graph
            The graph to analyze.
        attribute : str
            Node attribute on which to compute assortativity.

        Returns
        -------
        dict
            A dictionary containing computed statistics.
        """
        stats = {}
        
        # Diameter (only if connected)
        if nx.is_connected(graph):
            stats['diameter'] = nx.diameter(graph)
            stats['average_shortest_path_length'] = nx.average_shortest_path_length(graph)
        else:
            stats['diameter'] = None
            stats['average_shortest_path_length'] = None
        
        # Clustering coefficient
        stats['clustering_coefficient'] = nx.average_clustering(graph)
        
        # Homophily (assortativity) by attribute
        try:
            stats['homophily'] = nx.attribute_assortativity_coefficient(graph, attribute)
        except Exception as e:
            print(f"Error computing homophily: {e}")
            stats['homophily'] = np.nan
        
        return stats



    def compute_network_statistics(self, graph: nx.Graph, attribute: str) -> dict:
        """
        Compute exact network statistics via python-igraph:
           diameter
           average shortest-path length
           global clustering coefficient
           assortativity (homophily) on a node attribute

        Parameters
        ----------
        graph : nx.Graph
            The NetworkX graph to analyze.
        attribute : str
            Node attribute name for the assortativity coefficient.

        Returns
        -------
        stats : dict
            {
              'diameter': int,
              'average_shortest_path_length': float,
              'clustering_coefficient': float,
              'homophily': float
            }
        """
        # 1) Build an igraph.Graph from the NX graph
        #    (reindex nodes to 0..n-1)

        prefix = getattr(self, 'plot_prefix', '')

        nodes = list(graph.nodes())
        idx = {node: i for i, node in enumerate(nodes)}
        edges = [(idx[u], idx[v]) for u, v in graph.edges()]
        ig_g = ig.Graph(edges=edges, directed=False)

        # 2) Compute diameter and average shortest-path length
        diam = ig_g.diameter()  # exact
        avg_sp = ig_g.average_path_length(directed=False)  # exact

        # 3) Global clustering coefficient (transitivity)
        clust = ig_g.transitivity_undirected()

        degrees = [d for _, d in graph.degree()]
        avg_deg = sum(degrees) / graph.number_of_nodes()


        # 4) Assortativity (homophily) by attribute
        #    extract the attribute list in node order
        attr_vals = [graph.nodes[n].get(attribute) for n in nodes]
        # decide if numeric or nominal
        if all(isinstance(x, (int, float, np.floating)) for x in attr_vals):
            hom = ig_g.assortativity(attr_vals, directed=False)
        else:
            # map unique categories to integers
            cats = {v:i for i,v in enumerate(sorted(set(attr_vals)))}
            hom = ig_g.assortativity_nominal([cats[v] for v in attr_vals],
                                             directed=False)

        return {
            'label_id': prefix, 
            'diameter': diam,
            'average_shortest_path_length': avg_sp,
            'clustering_coefficient': clust,
            'homophily': hom, 
            'average_degree': avg_deg,
        }

    ############################################################################
    # ============================ VISUALIZATION ============================== #
    ############################################################################


    def plot_degree_distribution(self, graph: nx.Graph, 
                                 culture_value: int = None, 
                                 color: str = 'b', 
                                 title: str = "Degree Distribution") -> None:
        """
        Plots the degree distribution for nodes in a graph.
        Optionally filters nodes based on a specific culture value.
        """
        import matplotlib.pyplot as plt

        prefix = getattr(self, 'plot_prefix', '')

        if culture_value is not None:
            degree_sequence = [d for n, d in graph.degree() 
                               if graph.nodes[n].get('culture') == culture_value]
        else:
            degree_sequence = [d for _, d in graph.degree()]

        # Count occurrences of each degree
        degree_count = {x: degree_sequence.count(x) for x in degree_sequence}
        if not degree_count:
            print("No nodes or no degrees found.")
            return

        degree, count = zip(*sorted(degree_count.items()))
        
        plt.figure(figsize=(8, 5))
        plt.bar(degree, count, width=0.80, color=color)
        plt.title(title if culture_value is None else f"{title} (Culture {culture_value})")
        plt.ylabel("Count")
        plt.xlabel("Number of connections")
        plt.xticks(degree, degree)
        plt.savefig(f"{prefix}_degree_distribution_{culture_value if culture_value is not None else 'all'}_{ts}.png")
        plt.close()


    def visualize_graph_with_component_labels(self, graph: nx.Graph, 
                                              pos: dict, 
                                              node_colors: list,
                                              title: str = "Graph Visualization",
                                              node_size: int = 35,
                                              edge_color: str = 'gray',
                                              cmap=plt.cm.Blues) -> None:
        """
        Visualizes a graph with connected component labels indicating occupation and culture.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 12))
        nx.draw(graph, pos,
                with_labels=False,
                node_color=node_colors,
                edge_color=edge_color,
                node_size=node_size,
                cmap=cmap)
        plt.title(title)
        
        # Annotate connected components
        for component in nx.connected_components(graph):
            sample_node = list(component)[0]
            component_pos = pos[sample_node]
            occupation = graph.nodes[sample_node].get('occupation', 'N/A')
            c_val = graph.nodes[sample_node].get('culture')
            culture_str = "high" if c_val == 1 else "low"

            plt.text(
                component_pos[0],
                component_pos[1],
                f"{occupation}\n{culture_str}",
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.5)
            )
        plt.savefig(f"graph_visualization_{ts}.png")
        plt.close()


    def plot_degree_vs_attribute(self, graph: nx.Graph, 
                                 attribute: str,
                                 title: str = "Degree vs Attribute") -> None:
        """
        Plots node degree against a specified attribute value.
        """
        import matplotlib.pyplot as plt
        prefix = getattr(self, 'plot_prefix', '')

        degrees = [graph.degree(node) for node in graph.nodes]
        attrs = [graph.nodes[node].get(attribute, None) for node in graph.nodes]

        filtered_data = [(d, a) for d, a in zip(degrees, attrs) if a is not None]
        if not filtered_data:
            print(f"No valid data to plot for attribute '{attribute}'.")
            return

        degrees, attrs = zip(*filtered_data)
        
        plt.figure(figsize=(8, 5))
        plt.scatter(attrs, degrees, alpha=0.7, edgecolors='k')
        plt.title(title)
        plt.xlabel(attribute.capitalize())
        plt.ylabel("Degree")
        plt.grid(True)
        plt.savefig(f"{prefix}_degree_vs_{attribute}_{ts}.png")
        plt.close()


    def plot_degree_vs_attribute_colored(self, graph: nx.Graph,
                                         attribute: str,
                                         color_by: str,
                                         title: str = "Degree vs Attribute") -> None:
        """
        Plots node degree against a specified attribute value, 
        coloring points by another categorical attribute.
        """
        import matplotlib.pyplot as plt
        prefix = getattr(self, 'plot_prefix', '')

        degrees = [graph.degree(n) for n in graph.nodes]
        attrs = [graph.nodes[n].get(attribute, None) for n in graph.nodes]
        colors = [graph.nodes[n].get(color_by, None) for n in graph.nodes]

        filtered_data = [
            (deg, attr, col) for deg, attr, col 
            in zip(degrees, attrs, colors)
            if attr is not None and col is not None
        ]
        if not filtered_data:
            print(f"No valid data for plotting '{attribute}' vs '{color_by}'.")
            return

        degrees, attrs, colors = zip(*filtered_data)

        # Map unique values of 'color_by' to color indices
        unique_vals = sorted(set(colors))
        val_to_idx = {val: i for i, val in enumerate(unique_vals)}
        color_map = [plt.get_cmap('tab10')(val_to_idx[col]) for col in colors]

        plt.figure(figsize=(8, 5))
        plt.scatter(attrs, degrees, c=color_map, alpha=0.7, edgecolors='k', s=35)
        plt.title(title)
        plt.xlabel('Culture value')
        plt.ylabel("Number of connections")
        plt.grid(True)

        # Legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=plt.get_cmap('tab10')(i), markersize=8)
            for i in range(len(unique_vals))
        ]
        plt.legend(handles, unique_vals, title=color_by.capitalize(),
                   bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(f"{prefix}degree_vs_{attribute}_colored_by_{color_by}_{ts}.png")
        plt.close()

    ############################################################################
    # ========================== LAYOUT FUNCTION ============================== #
    ############################################################################

    @staticmethod
    def custom_layout(G: nx.Graph) -> dict:
        """
        Creates a custom layout by first using circular_layout,
        then shifting nodes belonging to each occupation slightly.
        """
        pos = nx.spring_layout(G)
        occupation_nodes = {}
        for node, data in G.nodes(data=True):
            occupation = data.get('occupation', 'N/A')
            if occupation not in occupation_nodes:
                occupation_nodes[occupation] = []
            occupation_nodes[occupation].append(node)
        
        # Slight displacement for each occupation group
        for occupation, nodes in occupation_nodes.items():
            displacement = 0.01 * (random.random() - 0.5)
            for node in nodes:
                pos[node][0] += displacement
                pos[node][1] += displacement
        return pos

    ############################################################################
    # ====================== HIGH-LEVEL WORKFLOW METHOD ======================= #
    ############################################################################

    def create_and_analyze_graph(self,
                                func : Callable, 
                                 m_high: int = 8,
                                 m_low: int = 4,
                                 p: float = 0.8,
                                 max_degree: int = 150,
                                 reorient_edges_p: float = 0.1,
                                 attribute_name: str = 'cult1', #not used directly in graph generation
                                 attribute_name2 : str =  'spending in Restaurants', #notused directly in graph generation
                                 visualize: bool = True) -> nx.Graph:
        """
        Creates a combined graph from the instance dataframe by partitioning 
        by (cult, HC24), building each subgraph with a Havel-Hakimi generator 
        (or a powerlaw generator), and merging them.
        
        Parameters
        ----------
        m_high : int
            Number of edges to attach for high-culture nodes.
        m_low : int
            Number of edges to attach for low-culture nodes.
        p : float
            Probability for triangle formation in powerlaw-based generator (unused in HH).
        max_degree : int
            Maximum degree for nodes.
        reorient_edges_p : float
            Fraction of edges to reorient in the final graph.
        attribute_name : str
            Column name to use for the Havel-Hakimi degree computation.
        visualize : bool
            Whether to produce plots.

        Returns
        -------
        nx.Graph
            The final merged graph.
        """
        # Copy the dataframe to avoid mutating the original
        df = self.dataframe.copy()
        df['occupation'] = df[self._col3]
        df['culture'] = df[self._col4]

        # Combined graph
        G_combined = nx.Graph()
        print('grouping starts')
        # Group by (culture, occupation) and create subgraphs
        for (culture, occupation), group in df.groupby(['culture', 'occupation']):
            print(culture, occupation)
            group_size = group.shape[0]
            sub_m = m_high if culture == 1 else m_low

            # Example: We use Havel-Hakimi for subgraph generation
            # (or optionally use powerlaw_cluster_graph_modified_with_attribute_and_maxdegree2)
            G_partial = self.havel_hakimi_graph_modified_with_attribute(group,
                                                                        func,
                                                                        seed=None)

            # Merge subgraph (disjoint_union to keep node IDs separate)
            G_combined = nx.disjoint_union(G_combined, G_partial)
        print("G_combined built")
        # Visualization steps
        if visualize:
            # Plot degree vs. attribute (colored by occupation)
            print("plot_degree_vs_attribute_colored")
            self.plot_degree_vs_attribute_colored(G_combined,
                                                  attribute=attribute_name,
                                                  color_by='occupation',
                                                  title="Degree vs Culture")

            # Plot degree distribution by culture
            print("plot_degree_distribution")
            self.plot_degree_distribution(G_combined, culture_value=1, color='b',
                                          title="Degree Distribution (Culture=1)")
            self.plot_degree_distribution(G_combined, culture_value=0, color='b',
                                          title="Degree Distribution (Culture=0)")

            # Layout and node-colors for occupation
            """
            print("custom layout building")
            pos = self.custom_layout(G_combined)
            unique_occupations = df['occupation'].unique()
            occupation_colors = {
                occ: plt.get_cmap('tab10')(i)
                for i, occ in enumerate(unique_occupations)
            }
            node_colors = [
                occupation_colors[G_combined.nodes[node]['occupation']]
                for node in G_combined.nodes
            ]
            print("visualize_graph_with_component_labels")
            # Visualize combined graph prior to reorientation
            
            self.visualize_graph_with_component_labels(
                graph=G_combined,
                pos=pos,
                node_colors=node_colors,
                title="Person Network (Before Edge Reorientation)"
            )
            """
        # Reorient edges
        print("doing reorient_edges_preserve_degree")
        self.reorient_edges_preserve_degree(G_combined, reorient_edges_p)

        # Visualize after reorientation
        if visualize:
            self.plot_degree_distribution(G_combined, None, color='b',
                                          title="Degree (the number of connections) Distribution (After Reorientation)")
            """
            self.visualize_graph_with_component_labels(
                graph=G_combined,
                pos=pos,
                node_colors=node_colors,
                title="Person Network (After Edge Reorientation)"
            )
            """
            # Plot again, in case the structure changed significantly
            self.plot_degree_vs_attribute_colored(G_combined,
                                                  attribute=attribute_name,
                                                  color_by='occupation',
                                                  title="Degree (Number of connections) vs Culture Value (After Reorientation)")

            # Compute and print stats
            culture_stats = self.compute_network_statistics(G_combined, 'culture')
            occupation_stats = self.compute_network_statistics(G_combined, 'occupation')
            print("Statistics by Culture:", culture_stats)
            print("Statistics by Occupation:", occupation_stats)
        self.graph = G_combined
        return G_combined
    


    def visualize_attr_vs_deg(self, G, attr : str):
        prefix = getattr(graph._analyzer, 'plot_prefix', '')

        degrees = []
        attribute_values = []

        # Iterate over nodes to collect degree and the corresponding attribute value.
        for node in G.nodes():
            degrees.append(G.degree[node])
            attribute_values.append(G.nodes[node][attr])

        # Create a scatter plot: x-axis for degree, y-axis for attribute value.
        plt.figure(figsize=(8, 5))
        plt.scatter(degrees, attribute_values, color='blue', alpha=0.7)
        plt.xlabel("Degree")
        plt.ylabel('denormalized total emission from food')
        plt.title("Degree vs denormalized total emission from food for Network Graph")
        plt.savefig(f"{prefix}degree_vs_{attr}_{ts}.png")
        plt.close()

    @staticmethod
    def func0(val, val2):
        return 3 * np.log(val + 10) + 2 + 0.0*np.log(0.0001*val2 + 1)


import random, time, os
from datetime import datetime
from typing import Callable, Optional

import networkx as nx
import numpy as np
import pandas as pd

# We *inherit* everything from the rich, data-driven SocialNetworkAnalyzer
# but replace the network-generation step with a plain Holme–Kim
# power-law-with-clustering model that ignores the input attributes
# when wiring the nodes.  All the analysis / plotting helpers are
# reused verbatim via inheritance.



class BaselineNetworkAnalyzer(SocialNetworkAnalyzer):
    """Baseline generator – same public API, but the network is built
    *without* any culture / occupation partitioning.  We simply grow a
    Holme-Kim power-law-cluster graph (``nx.powerlaw_cluster_graph``)
    until the number of nodes matches the input DataFrame.  Node
    attributes from the DataFrame are *still* copied onto the graph so
    that downstream AgentManipulator logic can run unchanged.
    """

    def __init__(self, dataframe: pd.DataFrame, seed: int = 1996):
        # The parent class expects four column names; they are irrelevant
        # for the baseline, so we pass placeholders.
        super().__init__(
            dataframe=dataframe,
            col="dummy1",  # unused
            col2="dummy2", # unused
            col3="HC24",   # still useful for visualisation, if present
            col4="cult",   # still useful for visualisation, if present
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Network generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _grow_powerlaw_cluster_graph(
        n: int,
        avg_deg: float = 12.0,
        p_triangle: float = 0.35,
        seed: Optional[int] = None,
    ) -> nx.Graph:
        """Return a Holme–Kim graph with *approximately* the requested
        average degree.

        Notes
        -----
        The Holme–Kim model attaches ``m`` edges for every newly added
        node; the final average degree is roughly ``2m``.  We therefore
        choose ``m = round(avg_deg/2)``.
        """
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        m = max(1, int(round(avg_deg / 2)))
        G = nx.powerlaw_cluster_graph(n=n, m=m, p=p_triangle, seed=rng)
        return G

    # ------------------------------------------------------------------
    # Public entry point – mirrors the original ``create_and_analyze_graph``
    # but calls the simplified generator above and skips culture/occupation
    # partitioning.
    # ------------------------------------------------------------------
    """
    def create_and_analyze_graph(
        self,
        avg_degree: float = 12.0,
        p_triangle: float = 0.35,
        reorient_edges_p: float = 0.10,
        visualize: bool = True,
    ) -> nx.Graph:
        Build a *purely structural* baseline network and run all the
        usual plots / stats.  All parameters mirror the parent class so
        you can drop this in without changing the driver script.

        Parameters
        ----------
        avg_degree : float
            Desired average degree (≈ 2·m in Holme–Kim terminology).
        p_triangle : float
            Triangle-formation probability in the Holme–Kim model.
        reorient_edges_p : float
            Fraction of edges to rewire after generation (degree-preserving).
        visualize : bool
            Produce the same diagnostic plots as the parent class.
      
        n = self.dataframe.shape[0]
        print(
            f"[baseline] generating Holme–Kim graph for {n:,} nodes, "
            f"avg_degree≈{avg_degree}, p_triangle={p_triangle}"
        )

        G = self._grow_powerlaw_cluster_graph(
            n=n,
            avg_deg=avg_degree,
            p_triangle=p_triangle,
            seed=random.randrange(1 << 30),
        )

        # --------------------------------------------------------------
        # Copy *all* DataFrame columns onto the nodes.
        # We keep the original row order but shuffle the mapping so that
        # node degree is NOT correlated with any attribute.
        # --------------------------------------------------------------
        shuffled_indices = list(self.dataframe.index)
        random.shuffle(shuffled_indices)
        for node, row_idx in zip(G.nodes(), shuffled_indices):
            G.nodes[node].update(self.dataframe.loc[row_idx].to_dict())

        # Save on the instance for later use (exactly as parent does)
        self.graph = G

        # --------------------------------------------------------------
        # Optional visualisation / stats
        # --------------------------------------------------------------
        if visualize:
            print("[baseline] plotting diagnostics …")
            self.plot_degree_distribution(G, None, color="steelblue", title="Degree distribution – baseline")
            if "denormalized total emission from food" in self.dataframe.columns:
                self.visualize_attr_vs_deg(G, "denormalized total emission from food")
            # global stats
            stats = self.compute_network_statistics(G, attribute="cult" if "cult" in G.nodes[0] else list(G.nodes()[0]).pop())
            print("[baseline] global statistics:", stats)

        return G
    """



    def powerlaw_cluster_graph_with_maxdegree(self,
        n: int,
        avg_deg: float = 12.0,
        p_triangle: float = 0.35,
        max_degree: int = 150,
        seed: Optional[int] = None
    ) -> nx.Graph:
        """
        Grow a Holme–Kim power-law cluster graph on n nodes with:
        - average degree ≈ 2·m where m = round(avg_deg/2)
        - triangle-closing probability p_triangle
        - maximum allowed degree capped at max_degree

        Parameters
        ----------
        n : int
            Number of nodes.
        avg_deg : float
            Desired average degree ≈ 2*m.
        p_triangle : float
            Probability to add a triangle on each attachment.
        max_degree : int
            Hard cap on any node’s degree.
        seed : int | None
            RNG seed (passed to random.Random if not None).

        Returns
        -------
        G : networkx.Graph
            An undirected graph on n nodes.
        """
        # choose m so that final average degree ~2*m
        m = max(1, int(round(avg_deg / 2)))

        # pick RNG
        rng = random.Random(seed) if seed is not None else random

        # start with m isolated nodes
        G = nx.empty_graph(m)
        # “repeated_nodes” stores endpoints for preferential attachment
        repeated_nodes = list(G.nodes())
        next_node = m

        while next_node < n:
            # filter out any nodes that have hit their max degree
            repeated_nodes = [u for u in repeated_nodes if G.degree(u) < max_degree]
            # if we don't have enough valid targets, stop early
            if len(repeated_nodes) < m:
                break

            # pick m distinct targets
            targets = rng.sample(repeated_nodes, m)
            # first attach to one
            first = targets.pop()
            G.add_edge(next_node, first)
            repeated_nodes.append(first)
            attached = 1

            # attach the remaining m-1 edges
            while attached < m:
                if rng.random() < p_triangle:
                    # try to close a triangle
                    nbrs = [
                        v for v in G.neighbors(first)
                        if v != next_node
                        and G.degree(v) < max_degree
                        and not G.has_edge(next_node, v)
                    ]
                    if nbrs:
                        v = rng.choice(nbrs)
                        G.add_edge(next_node, v)
                        repeated_nodes.append(v)
                        attached += 1
                        continue

                # else preferential attachment from remaining targets
                if targets:
                    u = targets.pop()
                    G.add_edge(next_node, u)
                    repeated_nodes.append(u)
                    attached += 1

            # add this new node m times for future preferential picks
            repeated_nodes.extend([next_node] * m)
            next_node += 1

        return G


    def create_and_analyze_graph(
        self,
        avg_degree: float = 14.0,
        p_triangle: float = 0.35,
        reorient_edges_p: float = 0.0,
        visualize: bool = True,
        capped = False, 
        tseed = None
        ) -> nx.Graph:
        """
        Build (once) a base Holme–Kim graph, then each call returns a
        copy, optionally rewired in a degree-preserving way.
        """

        # 1) build 
        n = self.dataframe.shape[0]
        print('powerlaw will start', datetime.now().strftime("%m%d_%H%M%S"))
        if capped:
            G = self.powerlaw_cluster_graph_with_maxdegree(
                n=n,
                avg_deg=avg_degree,
                p_triangle=p_triangle,
                max_degree = 50,
                seed=tseed,
            )
        else:
            G = self._grow_powerlaw_cluster_graph(
                n=n,
                avg_deg=avg_degree,
                p_triangle=p_triangle,
                seed=tseed,
            )
        print('powerlaw ends', datetime.now().strftime("%m%d_%H%M%S"))
        # shuffle & attach attributes exactly once
        shuffled = list(self.dataframe.index)
        random.shuffle(shuffled)
        for node, idx in zip(G.nodes(), shuffled):
            G.nodes[node].update(self.dataframe.loc[idx].to_dict())
        self._base_graph = G
        print('attr attached', datetime.now().strftime("%m%d_%H%M%S"))
        # 2) copy the cached graph and apply any rewiring
        G = self._base_graph.copy()
        if reorient_edges_p and reorient_edges_p > 0:
            self.reorient_edges_preserve_degree(G, reorient_edges_p)

        if visualize:
            # 1) Degree distribution
            self.plot_degree_distribution(G, culture_value=None,
                                          title="Baseline Degree Distribution")

            # 2) Degree vs. emission (if that attribute exists)
            if "denormalized total emission from food" in next(iter(G.nodes(data=True)))[1]:
                self.plot_degree_vs_attribute(
                    G,
                    attribute="denormalized total emission from food",
                    title="Degree vs Emission (Baseline)"
                )

            # 3) Global stats (e.g. assortativity on 'cult', if present)
            culture_stats    = self.compute_network_statistics(G, 'cult')   # instead of 'culture'
            occupation_stats = self.compute_network_statistics(G, 'HC24')   # instead of 'occupation'
            print("Statistics by Culture:", culture_stats)
            print("Statistics by Occupation:", occupation_stats)

        return G
    

